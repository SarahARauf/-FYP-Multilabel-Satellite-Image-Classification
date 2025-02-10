# Multilabel Multiband Satellite Image Classification Pipeline

## Overview
This repository contains a deep learning pipeline for classifying satellite images. This pipeline is based of the repository below, which includes the original pipeline from the paper "Residual Attention: A Simple but Effective Method for Multi-Label Recognition":https://github.com/Kevinz-code/CSRA

The pipeline leverages the CSRA (class-specific residual attention) module along with ResNet and ViT (vision transformer) backbones for the multilabel and multiband classification of satellite imagery.

## Features
- **Data Preprocessing**: The dataset used for this research is the Multiband BigEarthNet43 dataset. Due to the large size of our dataset, we are unable to upload it onto GitHub. Under 'usage' are the steps to preprocess the data yourself.
- **Modeling**: Integration of state-of-the-art deep learning models for image classification.
- **Evaluation**: Comprehensive evaluation metrics to assess model performance.
- **Visualization**: Tools for visualizing predictions and understanding model outputs.

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

## Getting Started

### Prerequisites
- Python 3.8 or higher
- GPU with CUDA support (optional but recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/satellite-image-classification.git
   cd satellite-image-classification
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   env\Scripts\activate    # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Data Preprocessing
  1. Download BigEarthNet S1 from this website and unzip: https://bigearth.net/#downloads
  2. Run ProcessLabels.py to create the csv file containing image path and labels for each image.
  3. Run Downsample.py to downsample the dataset to a fifth of the original size.
  4. Run Onehotencoded.py to split the dataset into train/val/test, apply one-hot encoding and turn into a .json file.
  5. Run ProcessImage.py to process the multiband satellite images into different 3 channel combinations.

```bash
python scripts/preprocess.py --input-dir data/raw --output-dir data/processed
```

#### Training
Train the model using the training script:
```bash
python scripts/train.py --data-dir data/processed --epochs 50 --batch-size 32
```

#### Evaluation
Evaluate the model performance:
```bash
python scripts/evaluate.py --model-path models/checkpoint.pth --data-dir data/processed
```

## Visualization
To visualize predictions, use the built-in tools provided in `notebooks/visualize_predictions.ipynb`.


## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- Satellite image datasets from [XYZ Provider].
