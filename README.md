# TinyVGG-inspired CNN Binary Classifier

This project implements a Convolutional Neural Network (CNN) binary classifier inspired by the TinyVGG architecture. The model is designed for binary classification tasks where the goal is to categorize input images into two classes.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

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

You can install the required packages using `pip`:

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/tinyvgg-binary-classifier.git
    cd tinyvgg-binary-classifier
    ```

2. Install the dependencies as mentioned above.

## Usage

1. **Data Preparation:**
   - Place your training and validation images in respective folders: `data/train/` and `data/val/`, with subfolders for each class (e.g., `class0/`, `class1/`).

2. **Train the Model:**
   - Run the training script with your dataset:
     ```bash
     python train.py --epochs 10 --batch-size 32 --lr 0.001
     ```

3. **Make Predictions:**
   - To test the model on a single image:
     ```bash
     python predict.py --image-path path_to_image.jpg
     ```

## Training

The training script supports the following arguments:

```bash
--epochs       # Number of epochs for training (default: 10)
--batch-size   # Batch size for training (default: 32)
--lr           # Learning rate (default: 0.001)
--data-dir     # Path to the dataset directory
```

Example command to start training:
```bash
python train.py --data-dir ./data --epochs 20 --batch-size 64 --lr 0.0001
```

## Evaluation

To evaluate the model on the validation dataset:

```bash
python evaluate.py --data-dir ./data/val
```

The script will output accuracy, loss, and other metrics for evaluation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
---
