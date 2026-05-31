import os
import shutil
import argparse
import sys


def check_dtd_structure(source_root):
    """
    检查DTD源目录结构是否完整
    """
    images_dir = os.path.join(source_root, 'images')
    labels_dir = os.path.join(source_root, 'labels')

    if not os.path.exists(images_dir):
        print(f"错误: 在 {source_root} 下未找到 'images' 文件夹。")
        return False, None, None

    if not os.path.exists(labels_dir):
        print(f"错误: 在 {source_root} 下未找到 'labels' 文件夹。")
        print("请确保你下载了完整的DTD数据集 (包含 labels 文件夹)。")
        return False, None, None

    return True, images_dir, labels_dir


def copy_files(file_list, source_images_dir, target_split_dir):
    """
    根据文件列表复制图片
    """
    count = 0
    total = len(file_list)

    print(f"正在处理 -> {os.path.basename(target_split_dir)} ({total} 张图片)")

    for rel_path in file_list:
        rel_path = rel_path.strip()  # 去除换行符，例如 "banded/banded_0001.jpg"

        # 构建源路径
        src_path = os.path.join(source_images_dir, rel_path)

        # 构建目标路径 (保留类别文件夹结构)
        # 例如: output/train/banded/banded_0001.jpg
        dst_path = os.path.join(target_split_dir, rel_path)

        if not os.path.exists(src_path):
            print(f"警告: 找不到源文件 {src_path}，跳过。")
            continue

        # 创建目标类别文件夹
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 复制文件
        shutil.copy2(src_path, dst_path)
        count += 1

        # 简单的进度显示
        if count % 100 == 0:
            sys.stdout.write(f"\r进度: {count}/{total}")
            sys.stdout.flush()

    print(f"\r完成: {count}/{total} 张图片已复制到 {target_split_dir}\n")


def process_fold(source_root, output_root, fold_num):
    """
    处理指定的 Fold (1-10)
    """
    valid, images_dir, labels_dir = check_dtd_structure(source_root)
    if not valid:
        return

    # 官方的三个数据集划分类型
    splits = ['train', 'val', 'test']

    print(f"========================================")
    print(f"正在生成 DTD 数据集 Fold {fold_num}")
    print(f"源目录: {source_root}")
    print(f"输出目录: {output_root}")
    print(f"========================================\n")

    for split in splits:
        # 构造官方 label 文件名，例如: train1.txt, val1.txt
        label_filename = f"{split}{fold_num}.txt"
        label_path = os.path.join(labels_dir, label_filename)

        if not os.path.exists(label_path):
            print(f"错误: 找不到官方划分文件 {label_path}")
            return

        # 读取图片列表
        with open(label_path, 'r') as f:
            file_list = f.readlines()

        # 定义输出路径，例如: output/train
        target_split_dir = os.path.join(output_root, split)

        # 如果目录存在，先清理（防止混合旧数据），或者你可以选择报错
        if os.path.exists(target_split_dir):
            print(f"清理旧目录: {target_split_dir}")
            shutil.rmtree(target_split_dir)
        os.makedirs(target_split_dir)

        # 执行复制
        copy_files(file_list, images_dir, target_split_dir)

    print("========================================")
    print(f"Fold {fold_num} 数据集划分完成！")
    print(f"请在训练代码中将 data_path 设置为: {output_root}")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DTD 数据集官方划分工具')

    # 参数设置
    parser.add_argument('--source', type=str, default=r'/root/TexRec/dtd',
                        help='原始 DTD 数据集路径 (必须包含 images 和 labels 文件夹)')
    parser.add_argument('--output', type=str, default=r'/root/TexRec/dtd_fold1',
                        help='划分后的数据集输出路径')
    parser.add_argument('--fold', type=int, default=1, choices=range(1, 11),
                        help='选择官方的第几种划分 (1-10)，默认为 1')

    args = parser.parse_args()

    process_fold(args.source, args.output, args.fold)