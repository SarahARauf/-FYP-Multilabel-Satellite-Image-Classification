# Multilabel Multiband Satellite Image Classification Pipeline

## Overview
This repository contains the deep learning pipeline for classifying satellite images used in the APSIPA ASC 2024 paper [Multi-band Satellite Image Analysis for Multi-label Classification](https://ieeexplore.ieee.org/document/10848859). This pipeline is based of the repository below, which includes the original pipeline from the paper [Residual Attention: A Simple but Effective Method for Multi-Label Recognition](https://github.com/Kevinz-code/CSRA)

The pipeline leverages the CSRA (class-specific residual attention) module along with ResNet and ViT (vision transformer) backbones for the multilabel and multiband classification of satellite imagery.

## Features
- **Data Preprocessing**: The dataset used for this research is the Multiband BigEarthNet43 dataset. Due to the large size of our dataset, we are unable to upload it onto GitHub. Under 'usage' are the steps to preprocess the data yourself.
- **Modeling**: Integration of the CSRA classifier with single-input and multi-input backbones architectures, including ResNet and ViT.
- **Evaluation**: Precision, Recall, F1 score, and mAP are used as performance metrics. Loss is calculated using Binary Cross Entropy.
- **Visualization**: Precision, recall and F1 score graphs, loss graph, class label sample size vs metrics graphs (imbalance plots).

## Pipeline Structure
```plaintext
├── data
│   ├── raw          # Raw satellite images
│   ├── processed    # Preprocessed images ready for modeling
├── notebooks         # Jupyter notebooks for exploratory analysis
├── models            # Saved models and checkpoints
├── scripts
│   ├── preprocess.py # Script for data preprocessing
│   ├── train.py      # Training script
│   ├── evaluate.py   # Evaluation script
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
```
## To-Do: Code Cleanup Tasks
### General Cleanup
- [ ] Remove unused imports and libraries.
- [ ] Add comments where necessary for better understanding.
- [ ] Remove all irrelevant comments.

## Getting Started

### Prerequisites
- Python 3.7
- CUDA 10.2
- Anaconda: Recommended for managing dependencies and creating a virtual environment

### Installation
1. Clone the repository:
   ```shell
   git clone https://github.com/your-username/satellite-image-classification.git
   cd satellite-image-classification
   ```
2. Create and activate a conda environment:
   ```shell
   conda env create -f environment.yml
   conda activate csra
   ```

### Usage

#### Data Preprocessing
  1. Download BigEarthNet S1 from this website and unzip: https://bigearth.net/#downloads
  2. Run ProcessLabels.py to create the csv file containing image path and labels for each image.
  3. Run Downsample.py to downsample the dataset to a fifth of the original size.
  4. Run Onehotencoded.py to split the dataset into train/val/test, apply one-hot encoding and turn into a .json file.
  5. Run ProcessImage.py to process the multiband satellite images into different 3 channel combinations.

#### Training
Train the model using the training script:
##### ResNet (Default)
```shell
python main.py --num_heads 1 --lam 0.1 --dataset bigearth --num_cls 43 --img_size 120 --total_epoch 100
```
##### ViT
```shell
python main.py --model vit_B16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset bigearth --num_cls 43 --total_epoch 100
```

#### Validation
Evaluate the model performance:

##### ResNet (Default)
```shell
python val.py --num_heads 1 --lam 0.1 --dataset bigearth --num_cls 43 --load_from path/to/checkpoint.pth --img_size 120
```

##### ViT
```shell
python val.py --num_heads 1 --lam 0.3 --dataset bigearth --num_cls 43 --load_from path/to/checkpoint.pth --img_size 224 --model vit_B16_224
```
#### Testing (Demo)
Demo the model on unseen data:

##### ResNet (Default)
```shell
python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset bigearth --load_from path/to/checkpoint.pth --img_dir utils/demo_test --img_size 120 --num_cls 43
```

##### ViT
```shell
python demo.py --model vit_B16_224 --num_heads 1 --lam 0.3 --dataset bigearth --load_from path/to/checkpoint.pth --img_dir utils/demo_test_rgb --img_size 224 --num_cls 43
```


#### Visualization

##### Training vs Validation Metrics
```shell
# Precision
python graph.py --csv_train path/to/train_results.csv --csv_val path/to/val_results.csv --output_dir path/to/output --title "Precision using Model" --y_label "precision"

# Recall
python graph.py --csv_train path/to/train_results.csv --csv_val path/to/val_results.csv --output_dir path/to/output --title "Recall using Model" --y_label "recall"

# F1 Score
python graph.py --csv_train path/to/train_results.csv --csv_val path/to/val_results.csv --output_dir path/to/output --title "F1 Score using Model" --y_label "f1"

# Loss
python graph.py --csv_train path/to/train_loss.csv --csv_val path/to/val_loss.csv --output_dir path/to/output --title "Loss using Model" --y_label "loss"
```

##### Class Imbalance Analysis
```shell
# Sample Size vs Metrics
python graph.py --csv_test path/to/class_results.csv --output_dir path/to/output --title "Sample Size vs Metrics" --x_label "Class Label Sample Size"
```


## Acknowledgments
- [PyTorch](https://pytorch.org/)
- Multi-band Satellite Image Analysis for Multi-label Classification: (https://ieeexplore.ieee.org/document/10848859)
- CSRA Official code: (https://github.com/Kevinz-code/CSRA/tree/master)
- BigEarthNet43 satellite image dataset: (https://bigearth.net/)
