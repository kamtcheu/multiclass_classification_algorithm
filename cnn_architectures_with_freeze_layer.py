#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, efficientnet_b0, vit_b_16
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%%
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#%%
# Data transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

loader_params = {
    #'batch_size': batch_size,
    'num_workers': 8  # increase this value to use multiprocess data loading
}
# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root="./", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./", train=False, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Function to load model
def get_model(model_name):
    global train_loader, test_loader, trainset, transform
    if model_name == "resnet18":
        model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, 10)
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(1280, 10)
    elif model_name == "vit_b_16":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Load CIFAR-10 dataset with new transform
        trainset = datasets.CIFAR10(root="./", train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root="./", train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=64, shuffle=True, **loader_params)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, **loader_params)

        model = vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(768, 10)
    else:
        raise ValueError("Model not supported")
    
    # Freeze all layers except the head
    for param in model.parameters():
        param.requires_grad = False

    # Ensure the new head's parameters are trainable
    if model_name == "resnet18":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "efficientnet_b0":
        for param in model.classifier[1].parameters():
            param.requires_grad = True
    elif model_name == "vit_b_16":
        for param in model.heads.head.parameters():
            param.requires_grad = True

    return model.to(device)

# Training function
def train_model(model, epochs=10):
    global model_name
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")
    
    end_time = time.time()
    torch.save(model.state_dict(), f'freez_fine_tuned_cnn_{model_name}.pth')
    print(f"Training time: {end_time - start_time:.2f} seconds")

# Evaluation function
def evaluate_model(model):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=trainset.classes))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=trainset.classes, yticklabels=trainset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Running experiments
for model_name in ["efficientnet_b0","resnet18", "vit_b_16"]:  # "resnet18", "efficientnet_b0", "vit_b_16"
    print(f"\nTraining {model_name}...")
    model = get_model(model_name)
    train_model(model, epochs=100)
    evaluate_model(model)