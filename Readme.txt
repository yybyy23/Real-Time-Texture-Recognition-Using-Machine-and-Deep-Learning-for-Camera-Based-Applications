Texture Recognition Project README
==================================

1. Project Overview
-------------------
This project contains Python scripts for texture image classification experiments. The code trains and evaluates several baseline and comparison models on texture datasets, mainly:

- DTD (Describable Textures Dataset)
- KTH-TIPS2-b texture dataset

Implemented experiment scripts include ResNet-18, MobileNetV2, ViT-lite, Swin Transformer, Bilinear CNN, CNN + SVM, teacher model training, DeiT-style knowledge distillation, and FPS benchmarking.


2. Environment Requirements
---------------------------
Recommended environment:

- Python 3.8 or later
- PyTorch
- torchvision
- timm
- scikit-learn
- numpy
- matplotlib
- seaborn
- pandas
- Pillow
- tqdm
- joblib

A typical installation command is:

pip install torch torchvision timm scikit-learn numpy matplotlib seaborn pandas pillow tqdm joblib

If CUDA is available, the scripts will automatically use GPU. Otherwise, they will run on CPU.


3. Project Files
----------------
Main scripts included in this project:

- train.py: ResNet-18 training and evaluation on DTD fold data.
- MobileNetV2.py: MobileNetV2 training and evaluation on DTD fold data.
- MobileNetV2-kth.py: MobileNetV2 training and evaluation on KTH-TIPS2-b.
- VIT-lite-DTD.py: ViT-lite training and evaluation on DTD fold data.
- VIT-lite.py: ViT-lite training and evaluation on KTH-TIPS2-b.
- swin_dtd384.py: Swin Transformer training and evaluation on DTD.
- swin_kth.py: Swin Transformer training and evaluation on KTH-TIPS2-b at 224 x 224 input size.
- swin_kth384.py: Swin Transformer training and evaluation on KTH-TIPS2-b at 384 x 384 input size.
- B_cnn_dtd.py: Bilinear CNN training and evaluation on DTD.
- B_cnn_kth.py: Bilinear CNN training and evaluation on KTH-TIPS2-b.
- cnn-kth.py: ResNet-based CNN baseline on KTH-TIPS2-b.
- cnn-svm.py: ResNet50 feature extractor + Linear SVM on DTD fold data.
- cnn-svm-kth.py: ResNet50 feature extractor + Linear SVM on KTH-TIPS2-b.
- train_teacher.py: ResNet50 teacher model training for knowledge distillation on KTH-TIPS2-b.
- DeiT.py: DeiT-style student model training using the trained teacher model.
- benchmark_fps.py: FPS and latency benchmark for MobileNet, Swin, and CNN+SVM models.
- divide.py: Utility script for generating official DTD train/validation/test folds.
- load.py: Simple dataset loading test script.


4. Data Used
------------
This project uses the following public texture datasets.

4.1 DTD: Describable Textures Dataset

Official website:
https://www.robots.ox.ac.uk/~vgg/data/dtd/

Direct dataset download:
https://www.robots.ox.ac.uk/~vgg/data/dtd/dtd-r1.0.1.tar.gz

The DTD dataset contains 47 texture categories. The official release includes image files and label split files. In this project, DTD is used in two formats:

A. Raw DTD format, used by scripts such as B_cnn_dtd.py and swin_dtd384.py:

/root/TexRec/dtd/
    images/
        banded/
        blotchy/
        braided/
        ...
    labels/
        train1.txt
        val1.txt
        test1.txt
        ...

B. Folded ImageFolder format, used by scripts such as train.py, MobileNetV2.py, VIT-lite-DTD.py, and cnn-svm.py:

/root/TexRec/dtd_fold1/
    train/
        class_1/
        class_2/
        ...
    val/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/
        ...

To generate the folded DTD structure from the official DTD release, run:

python divide.py --source /path/to/dtd --output /path/to/dtd_fold1 --fold 1

Then update the DATA_PATH variable in the corresponding scripts if your dataset is not stored at /root/TexRec/dtd_fold1.


4.2 KTH-TIPS2-b

Official KTH-TIPS download page:
https://www.csc.kth.se/cvap/databases/kth-tips/download.html

General KTH-TIPS homepage:
https://www.csc.kth.se/cvap/databases/kth-tips/index.html

This project uses KTH-TIPS2-b with 11 texture classes and four physical samples per class. The expected folder structure is:

/root/TexRec/KTH-TIPS2-b/
    aluminium_foil/
        sample_a/
        sample_b/
        sample_c/
        sample_d/
    brown_bread/
        sample_a/
        sample_b/
        sample_c/
        sample_d/
    corduroy/
        sample_a/
        sample_b/
        sample_c/
        sample_d/
    ...

The KTH scripts generally use sample_a, sample_b, and sample_c for training, and sample_d for validation or testing.


5. Important Path Configuration
-------------------------------
Most scripts use a hard-coded DATA_PATH variable near the top of the file. Before running any experiment, open the target script and change DATA_PATH to the actual dataset location on your machine.

Examples:

For DTD fold data:
DATA_PATH = "/root/TexRec/dtd_fold1"

For raw DTD images:
DATA_PATH = "/root/TexRec/dtd/images"

For KTH-TIPS2-b:
DATA_PATH = "/root/TexRec/KTH-TIPS2-b"

You can either modify these paths in the scripts or place the datasets under the same default directories.


6. How to Run the Code
----------------------
Step 1: Unzip the project files and enter the project directory.

unzip TexRec.zip
cd TexRec

Step 2: Install dependencies.

pip install torch torchvision timm scikit-learn numpy matplotlib seaborn pandas pillow tqdm joblib

Step 3: Download and prepare the datasets.

For DTD:
- Download dtd-r1.0.1.tar.gz from the official DTD website.
- Extract it to a local folder.
- Use divide.py to create the train/val/test folder structure if needed.

Example:

python divide.py --source /root/TexRec/dtd --output /root/TexRec/dtd_fold1 --fold 1

For KTH-TIPS2-b:
- Download KTH-TIPS2-b from the official KTH-TIPS download page.
- Extract it and make sure the folder structure follows class/sample_a, class/sample_b, class/sample_c, and class/sample_d.

Step 4: Run the desired experiment.

Examples for DTD:

python train.py
python MobileNetV2.py
python VIT-lite-DTD.py
python swin_dtd384.py
python B_cnn_dtd.py
python cnn-svm.py --data-path /root/TexRec/dtd_fold1 --batch-size 32 --save-bundle cnn_svm_bundle.pkl

Examples for KTH-TIPS2-b:

python cnn-kth.py
python MobileNetV2-kth.py
python VIT-lite.py
python swin_kth.py
python swin_kth384.py
python B_cnn_kth.py
python cnn-svm-kth.py

Example for teacher-student distillation:

python train_teacher.py
python DeiT.py

Note: DeiT.py requires a trained teacher checkpoint named kth_resnet50_best.pth by default. This file is generated by train_teacher.py.


7. FPS Benchmark
----------------
After training and saving the required checkpoints, benchmark inference speed with:

python benchmark_fps.py --device cpu --batch-size 1 --warmup 30 --runs 100 --mobilenet-ckpt texture_classifier_mobilenet.pth --swin-ckpt swin_dtd_384_best.pth --cnn-svm-bundle cnn_svm_bundle.pkl --num-classes 47

If you want to include preprocessing time using a real image, add:

--sample-image /path/to/sample_image.jpg

Example:

python benchmark_fps.py --device cpu --batch-size 1 --sample-image /root/TexRec/dtd_fold1/test/banded/banded_0001.jpg

For mobile or real-time deployment discussion, CPU benchmarking with batch_size = 1 is usually the most meaningful setting.


8. Output Files
---------------
Depending on the script, the code may generate:

- Trained model checkpoints, such as:
  - texture_classifier_resnet18.pth
  - texture_classifier_mobilenet.pth
  - kth_mobilenet_best.pth
  - kth_resnet18_best.pth
  - kth_resnet50_best.pth
  - dtd_vit_best.pth
  - kth_vit_best.pth
  - swin_dtd_384_best.pth
  - swin_best.pth
  - swin_384_best.pth
  - bcnn_dtd_best.pth
  - bcnn_best.pth
  - vit_distilled_best.pth

- Training curves, such as:
  - training_results_resnet18_optimized.png
  - training_results_mobilenet_v2.png
  - kth_mobilenet_history.png
  - kth_result.png
  - dtd_vit_history.png

- Confusion matrices, such as:
  - confusion_matrix_resnet18.png
  - confusion_matrix_mobilenet.png
  - kth_mobilenet_confusion.png
  - dtd_vit_confusion.png
  - kth_vit_confusion.png
  - Swin_DTD_384_confusion_matrix.png
  - Swin_confusion_matrix.png
  - Swin_384_confusion_matrix.png
  - BCNN_DTD_confusion_matrix.png
  - BCNN_confusion_matrix.png
  - svm_confusion_matrix.png
  - kth_svm_confusion_metrics.png

- CNN+SVM bundle:
  - cnn_svm_bundle.pkl


9. Reproducibility Notes
------------------------
- The code automatically uses CUDA if a GPU is available.
- Some scripts use random data augmentation, random split, Mixup, or random initialization, so results may vary slightly across runs.
- DTD experiments should preferably use the official train/val/test split generated by divide.py for fair comparison.
- KTH-TIPS2-b experiments in this code usually follow a sample-level split, where sample_a, sample_b, and sample_c are used for training and sample_d is used for validation or testing.
- If your folder names are different from those expected by the scripts, update either the folder names or the class list in the code.


10. Common Problems
-------------------
Problem: FileNotFoundError for DATA_PATH.
Solution: Check whether DATA_PATH in the script points to the correct dataset folder.

Problem: DeiT.py cannot find kth_resnet50_best.pth.
Solution: Run train_teacher.py first, or change TEACHER_PATH in DeiT.py to the correct teacher checkpoint path.

Problem: CUDA out of memory.
Solution: Reduce BATCH_SIZE near the top of the script. For 384 x 384 input models, batch size 16 or smaller may be safer.

Problem: timm is missing.
Solution: Install it with:

pip install timm

Problem: The number of classes does not match the checkpoint.
Solution: Make sure that the dataset path, class folders, and num_classes setting are consistent with the checkpoint being loaded.
