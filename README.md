# TinyVGG-inspired CNN Binary Classifier

This project implements a Convolutional Neural Network (CNN) binary classifier inspired by the TinyVGG architecture. The model is designed for binary classification tasks where the goal is to categorize input images into two classes.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Contributor](#contributor)

## Model Architecture

The CNN model is based on the TinyVGG architecture with modifications for binary classification. The architecture consists of:
- Multiple convolutional layers followed by max-pooling.
- ReLU activation functions after each convolution.
- Fully connected layers at the end to output a binary classification decision.

**Key Features:**
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

You can install the required packages using `pip`:

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

## Usage

1. **Data Preparation:**
   - Place your training and validation images in respective folders: `datasets/train/` and `datasets/val/`, with subfolders for each class (e.g., `class0/`, `class1/`).

2. **Dataset:**
   - For this project, you can use the dataset from Kaggle: [Binary Image Classification Dataset](https://www.kaggle.com/datasets/hasnainkhan0123/binary-image-classification)

## Training and Evaluation

To the train, evaluate and test the model, make necessary changes and run the main script.

   - The script will output accuracy, loss, and other metrics for evaluation.

## Contributor

This project is developed by B M Manohara @Manohara-Ai

---
