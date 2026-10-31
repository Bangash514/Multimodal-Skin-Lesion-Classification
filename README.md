# Multimodal-Skin-Lesion-Classification

Deep-learning experiments for **skin lesion classification** on the public **HAM10000** dermatoscopic dataset (7 diagnostic classes). The repository compares several CNN backbones, and includes experiments that combine the image with clinical metadata (age, sex, lesion localization).

> Research / learning code. Not a medical device and not for clinical use.

## Dataset
- **HAM10000** (Tschandl et al., *Scientific Data*, 2018): about 10,000 dermatoscopic images in 7 classes (akiec, bcc, bkl, df, mel, nv, vasc), with metadata (`age`, `sex`, `localization`, `dx`, `lesion_id`).
- The data are **not included** in this repository. Download them from the original source and check the licence terms before use.

## What is in this repository
All scripts are TensorFlow/Keras unless noted.

| Backbone | Script | Input |
|---|---|---|
| Custom CNN | `CustomCNN.py`, `Custom-Cnn` | image + metadata |
| VGG16 | `VGG16.py` | image + metadata |
| DenseNet121 | `DenseNet121.py`, `DL_DenseNet121` | image + metadata |
| DenseNet169 | `DenseNet169_ModelwithEnhancedMetrics.py` | image + metadata |
| EfficientNetB0 / B4 | `EfficientNetB0.py`, `EfficientNetB4` | image + metadata |
| InceptionResNetV2 | `InceptionResNetV2*` (several versions) | image only or image + metadata (varies by version) |
| InceptionV3 (GoogleNet) | `GoogleNet` | image only |
| ResNet50 (lightweight, PyTorch) | `ResNet.py` | image only |

Typical training setup in the scripts: ImageNet-pretrained backbone with partial fine-tuning, data augmentation, Adam optimiser, early stopping and checkpointing, with an image branch (global average pooling) and, where used, a small dense branch for the metadata, concatenated before the classifier.

## How to run
1. Install: `pip install tensorflow pandas numpy scikit-learn matplotlib` (plus `torch` for `ResNet.py`).
2. Download HAM10000 and put the images in class sub-folders as expected by `flow_from_directory`.
3. Edit the path variables at the top of the script (`base_dir`, `metadata_path`, ...), then run it.

## Results
| Model | Input | Validation accuracy | Notes |
|---|---|---|---|
| _TODO_ | _TODO_ | _TODO_ | _TODO_ |

Add the numbers from your own runs, and state the split used. Report macro-F1 or balanced accuracy as well as accuracy, because the nv class makes up about two thirds of HAM10000.

## Known issues and next steps
- **Metadata alignment:** in several image + metadata scripts the metadata batch is not matched row-by-row to the images in that batch. Results from those scripts should therefore be read as image-dominated, and should not be used to claim a benefit from metadata fusion. A corrected, aligned data pipeline is planned.
- **Validation split:** images are split at random, but HAM10000 has several images per lesion (`lesion_id`). A lesion-level split is needed to avoid optimistic accuracy.
- **Class imbalance:** report per-class metrics and use class weights or resampling.
- **Cleanup:** consistent file names, one config file for paths, and a requirements file.

## Authorship and use
The code and experiments in this repository were written and run by **BEN** (GitHub: [Bangash514](https://github.com/Bangash514)). The HAM10000 dataset is public and is not owned by the author; please follow its own licence.

If you use or adapt this code, please give credit by citing this repository and the author's name. For questions, collaboration or other uses, contact the author through GitHub (open an issue or send a message from the profile).

This code is released under the MIT licence (see `LICENSE`), which requires keeping the copyright notice.

## References
- Tschandl P., Rosendahl C., Kittler H., *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*, Scientific Data, 2018.

## Author
BEN, Software Engineer. GitHub: [Bangash514](https://github.com/Bangash514)
