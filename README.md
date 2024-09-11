# TinyVGG-inspired CNN Binary Classifier

This project implements a Convolutional Neural Network (CNN) binary classifier inspired by the TinyVGG architecture. The model is designed for binary classification tasks where the goal is to categorize input images into two classes.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training and Evaluation](#train-the-model-and-evalution-phase)
- [Contributor](#contributor)

## Model Architecture

The CNN model is based on the TinyVGG architecture with modifications to suit binary classification tasks. The architecture consists of:
- Multiple convolutional layers followed by max-pooling.
- ReLU activation functions after each convolution.
- Fully connected layers at the end to output a binary classification decision.

Key Features:
- Lightweight model, suitable for small datasets.
- Two output nodes using softmax activation for binary classification.

### Architecture Overview:
```
Conv2D -> ReLU -> MaxPooling
Conv2D -> ReLU -> MaxPooling
Flatten -> Fully Connected -> ReLU
Fully Connected -> Output (Softmax for binary classification)
```

## Requirements

To run this project, the following packages are required:
- Python 3.x
- PyTorch
- OpenCV (for image preprocessing)
- NumPy
- Matplotlib (for visualizations)
- tqdm (for progress tracking)

## Usage

1. **Data Preparation:**
   - Place your training and validation images in respective folders: `datasets/train/` and `test_dataset/`, with subfolders for each class (e.g., `class0/`, `class1/`).
  
For this project, you can use the dataset from Kaggle: [Binary Image Classification Dataset](https://www.kaggle.com/datasets/hasnainkhan0123/binary-image-classification)

## Train the Model and Evaluation Phase:
   - Run the main script with your datasets loaded, the script trains and evaluates and make predictions:

The script will also output accuracy, loss, and other metrics for evaluation, thanks to the helper functions.

## Contributor

This project is developed by B M Manohara @Manohara-Ai
---
