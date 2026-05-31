import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import timm
import joblib


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SVMInferenceWrapper:
    """
    将 CNN+SVM 封装成统一的 predict(x) 接口，方便和纯深度学习模型一起测速。
    这里测速的是：CNN 特征提取 + 标准化 + SVM 预测 的完整推理时间。
    """
    def __init__(self, feature_extractor, scaler, svm_clf, device):
        self.feature_extractor = feature_extractor.to(device)
        self.feature_extractor.eval()
        self.scaler = scaler
        self.svm_clf = svm_clf
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        feat = self.feature_extractor(x)
        feat = torch.flatten(feat, 1)
        feat = feat.cpu().numpy()
        feat = self.scaler.transform(feat)
        pred = self.svm_clf.predict(feat)
        return pred



def build_mobilenet(num_classes=47, checkpoint_path=None, device='cpu'):
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"MobileNet checkpoint not found: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model



def build_swin(num_classes=47, checkpoint_path=None, device='cpu'):
    model = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=num_classes)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Swin checkpoint not found: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model



def build_cnn_svm(bundle_path=None, device='cpu'):
    if not bundle_path or not os.path.exists(bundle_path):
        raise FileNotFoundError(f"CNN+SVM bundle not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    scaler = bundle['scaler']
    svm_clf = bundle['svm']

    # 这里保持与你 cnn-svm.py 一致：ResNet50 去掉 fc，仅作特征提取
    resnet = models.resnet50(weights=None)
    modules = list(resnet.children())[:-1]
    feature_extractor = nn.Sequential(*modules)

    # 如果你后续想换成自己微调后的CNN特征提取器，也可以在这里加载权重
    if bundle.get('feature_extractor_state_dict') is not None:
        feature_extractor.load_state_dict(bundle['feature_extractor_state_dict'], strict=False)

    return SVMInferenceWrapper(feature_extractor, scaler, svm_clf, device)



def get_transform(model_name):
    if model_name == 'mobilenet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'swin':
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'cnn_svm':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")



def get_dummy_input(model_name, batch_size, device):
    shape_map = {
        'mobilenet': (batch_size, 3, 224, 224),
        'swin': (batch_size, 3, 384, 384),
        'cnn_svm': (batch_size, 3, 224, 224),
    }
    return torch.randn(*shape_map[model_name], device=device)



def benchmark_one_model(model_name, model, batch_size=1, warmup=50, runs=200, device='cpu'):
    x = get_dummy_input(model_name, batch_size, device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    # 正式测速
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    avg_latency = total_time / runs / batch_size
    fps = (runs * batch_size) / total_time
    return avg_latency, fps



def benchmark_with_preprocess(model_name, model, image_path, warmup=20, runs=100, device='cpu'):
    transform = get_transform(model_name)
    image = Image.open(image_path).convert('RGB')

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            x = transform(image).unsqueeze(0).to(device)
            _ = model(x)
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            x = transform(image).unsqueeze(0).to(device)
            _ = model(x)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    avg_latency = total_time / runs
    fps = runs / total_time
    return avg_latency, fps



def print_result_table(results, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'Model':<15}{'Latency (ms/img)':>20}{'FPS':>15}")
    print("-" * 72)
    for name, latency, fps in results:
        print(f"{name:<15}{latency * 1000:>20.2f}{fps:>15.2f}")
    print("=" * 72)



def main():
    parser = argparse.ArgumentParser(description='Benchmark MobileNet / Swin / CNN+SVM FPS')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='测速设备；比较移动端建议用 cpu')
    parser.add_argument('--batch-size', type=int, default=1, help='建议 batch_size=1，更接近移动端单张推理')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--mobilenet-ckpt', type=str, default='texture_classifier_mobilenet.pth')
    parser.add_argument('--swin-ckpt', type=str, default='swin_dtd_384_best.pth')
    parser.add_argument('--cnn-svm-bundle', type=str, default='cnn_svm_bundle.pkl')
    parser.add_argument('--num-classes', type=int, default=47)
    parser.add_argument('--sample-image', type=str, default=None, help='若提供，则额外统计“预处理+推理”的端到端 FPS')
    args = parser.parse_args()

    set_seed(42)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('你指定了 cuda，但当前环境没有可用 GPU。')

    device = args.device
    print(f'当前测速设备: {device}')
    print('说明：为了比较移动端潜力，最有参考价值的是 CPU + batch_size=1。')

    mobilenet_model = build_mobilenet(num_classes=args.num_classes,
                                      checkpoint_path=args.mobilenet_ckpt,
                                      device=device)
    swin_model = build_swin(num_classes=args.num_classes,
                            checkpoint_path=args.swin_ckpt,
                            device=device)
    cnn_svm_model = build_cnn_svm(bundle_path=args.cnn_svm_bundle,
                                  device=device)

    pure_results = []
    for model_name, model in [('mobilenet', mobilenet_model), ('swin', swin_model), ('cnn_svm', cnn_svm_model)]:
        latency, fps = benchmark_one_model(model_name, model,
                                           batch_size=args.batch_size,
                                           warmup=args.warmup,
                                           runs=args.runs,
                                           device=device)
        pure_results.append((model_name, latency, fps))

    print_result_table(pure_results, '纯推理测速结果（不含磁盘读取）')

    if args.sample_image is not None:
        e2e_results = []
        for model_name, model in [('mobilenet', mobilenet_model), ('swin', swin_model), ('cnn_svm', cnn_svm_model)]:
            latency, fps = benchmark_with_preprocess(model_name, model,
                                                     image_path=args.sample_image,
                                                     warmup=max(10, args.warmup // 2),
                                                     runs=args.runs,
                                                     device=device)
            e2e_results.append((model_name, latency, fps))
        print_result_table(e2e_results, '端到端测速结果（预处理 + 推理）')

    print('\n结论建议：')
    print('1. 论文里主表建议报告 CPU / batch_size=1 / 不含I/O 的 FPS。')
    print('2. 作为补充，可再给一组“预处理+推理”的端到端 FPS。')
    print('3. 若目标是移动端部署，通常 FPS 更高、输入分辨率更小、参数更少的模型更有优势。')


if __name__ == '__main__':
    main()
