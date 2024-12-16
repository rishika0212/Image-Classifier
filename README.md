# Image-Classifier
A Python script for binary image classification using TensorFlow and MobileNetV2. 
A Python project for binary image classification using TensorFlow and MobileNetV2. The script handles dataset cleaning, data augmentation, transfer learning, and evaluation with key metrics such as precision, recall, and accuracy.

Features
Dataset Validation: Automatically removes invalid or corrupt images.
Data Augmentation: Enhances training by applying random flips, rotations, and zooms.
Transfer Learning: Uses MobileNetV2 pre-trained on ImageNet for efficient model building.
Model Evaluation: Provides detailed metrics including precision, recall, and accuracy.

# Image Classification with TensorFlow

This project demonstrates a binary image classification pipeline using TensorFlow and MobileNetV2.

## Features
- Dataset validation and cleaning
- Data augmentation
- Transfer learning with MobileNetV2
- Training and evaluation with metrics like Precision, Recall, and Accuracy

## How to Use
1. Place your dataset in the `data` directory, organized by class folders.
2. Run the script to train the model and evaluate it on a test dataset.
3. Modify the script for custom thresholds or datasets if needed.

## Requirements
- TensorFlow
- OpenCV
- Matplotlib
- NumPy

Install dependencies:
```bash
pip install tensorflow opencv-python matplotlib numpy


Run the script:
python image-classifier.py
Training logs will be saved in the logs/ directory. View them using TensorBoard:
tensorboard --logdir=logs
Adjust the threshold or dataset paths as needed in the script.

Results
The model outputs:
Training and validation loss and accuracy plots.
Metrics: Precision, Recall, and Accuracy on the test set.
Prediction on a new test image.

Example Prediction
Load a test image (happytest.jpg) and predict its class using the trained model. The script provides a raw prediction score and classifies it based on a threshold.
