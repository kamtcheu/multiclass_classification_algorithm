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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
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

# Store results
results = {}

# Train and evaluate SVM
def train_evaluate_svm():
    print("\nTraining SVM...")
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    acc = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    results['SVM'] = {'accuracy': acc, 'f1_score': f1}
    print("SVM Classification Report:")
    print(classification_report(test_labels, predictions))
    logging.info("SVM Classification Report:")
    logging.info(classification_report(test_labels, predictions))

# Train and evaluate Random Forest
def train_evaluate_rf():
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    acc = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    results['Random Forest'] = {'accuracy': acc, 'f1_score': f1}
    print("Random Forest Classification Report:")
    print(classification_report(test_labels, predictions))
    logging.info("Random Forest Classification Report:")
    logging.info(classification_report(test_labels, predictions))

# Train and evaluate CNN models
def evaluate_model_metrics(model, model_name):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    results[model_name] = {'accuracy': acc, 'f1_score': f1}

# Running experiments
#for model_name in ["resnet18", "efficientnet_b0", "vit_b_16"]:
#    print(f"\nTraining {model_name}...")
#    model = get_model(model_name)
#    train_model(model, model_name, epochs=5)
#    evaluate_model_metrics(model, model_name)

# Train and evaluate classical ML models
train_evaluate_svm()
train_evaluate_rf()
#%%
test_set_len=10000
results = {
    'SVM': {'accuracy': 0.46, 'inf_time': 6*3600/test_set_len}, #6h
    'Random Forest': {'accuracy': 0.4, 'inf_time':10*60/test_set_len},#10 min
    'resnet18': {'accuracy': 0.78, 'inf_time': 9806/test_set_len}, #~2h45
    'efficientnet_b0': {'accuracy': 0.82, 'inf_time': 1550/test_set_len},  #~25 min
    'vit_b_16': {'accuracy': 0.56, 'inf_time': 51174/test_set_len} #~14h
}
# Plot comparison
plt.figure(figsize=(10, 6))
x_labels = list(results.keys())
accuracy_vals = [results[m]['accuracy'] for m in x_labels]
f1_vals = [results[m]['inf_time'] for m in x_labels]

x = np.arange(len(x_labels))
width = 0.35
plt.bar(x - width/2, accuracy_vals, width, label='Accuracy')
plt.bar(x + width/2, f1_vals, width, label='Inference Time')

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.xticks(x, x_labels, rotation=45)
plt.legend()
plt.show()

# %%
