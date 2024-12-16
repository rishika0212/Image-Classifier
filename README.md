# Image-Classifier
A Python script for binary image classification using TensorFlow and MobileNetV2. 
A Python project for binary image classification using TensorFlow and MobileNetV2. The script handles dataset cleaning, data augmentation, transfer learning, and evaluation with key metrics such as precision, recall, and accuracy.

Features
Dataset Validation: Automatically removes invalid or corrupt images.
Data Augmentation: Enhances training by applying random flips, rotations, and zooms.
Transfer Learning: Uses MobileNetV2 pre-trained on ImageNet for efficient model building.
Model Evaluation: Provides detailed metrics including precision, recall, and accuracy.

Project Structure
├── image-classifier.py  # Main script
├── data/                # Directory for dataset (not included in the repo)
├── logs/                # TensorBoard logs
├── README.md            # Project documentation
└── .gitignore           # Files and directories to exclude

Prerequisites:
Python 3.8+
TensorFlow
OpenCV
Matplotlib
NumPy

Install dependencies using:
pip install tensorflow opencv-python matplotlib numpy

Usage
Organize your dataset in the data/ directory with subfolders for each class.

Example:
data/
├── happy/
└── sad/

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
