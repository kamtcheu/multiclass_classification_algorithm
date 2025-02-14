import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, efficientnet_b0, vit_b_16
import time
import os
import logging
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib


# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logging.info(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root="./", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./", train=False, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)


# Extract features for classical ML models
def extract_features(dataloader):
    features, labels = [], []
    for images, lbls in dataloader:
        images = images.view(images.size(0), -1).numpy()
        features.extend(images)
        labels.extend(lbls.numpy())
    return np.array(features), np.array(labels)

train_features, train_labels = extract_features(train_loader)
test_features, test_labels = extract_features(test_loader)

# Train and evaluate SVM
def train_evaluate_svm():
    print("\nTraining SVM...")
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(train_features, train_labels)
    # Save the trained SVM model
    joblib.dump(clf, "svm_model.pkl")
    print("Model saved successfully!")

    predictions = clf.predict(test_features)
    print("SVM Classification Report:")
    print(classification_report(test_labels, predictions))
    logging.info("SVM Classification Report:")
    logging.info(classification_report(test_labels, predictions))

# Train and evaluate Random Forest
def train_evaluate_rf():
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_features, train_labels)
    # Save the trained Random Forest model
    joblib.dump(clf, "random_forest_model.pkl")
    print("Model saved successfully!")

    predictions = clf.predict(test_features)
    print("Random Forest Classification Report:")
    print(classification_report(test_labels, predictions))
    logging.info("Random Forest Classification Report:")
    logging.info(classification_report(test_labels, predictions))
#%%
# Running experiments
"""
for model_name in ["resnet18", "efficientnet_b0", "vit_b_16"]:
    print(f"\nTraining {model_name}...")
    model = get_model(model_name)
    train_model(model, model_name, epochs=5)
    evaluate_model(model)
"""
# Train and evaluate classical ML models
#train_evaluate_svm()
train_evaluate_rf()
