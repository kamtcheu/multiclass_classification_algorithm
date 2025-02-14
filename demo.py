# Import necessary libraries
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
# Load trained models
rf_model = joblib.load("random_forest_model.pkl")  # Load Random Forest
#svm_model = joblib.load("svm_model.pkl")  # Load SVM
efnet = models.efficientnet_b0(num_classes=10)
efnet.load_state_dict(torch.load("fine_tuned_cnn_efficientnet_b0.pth", map_location=torch.device('cpu')))

#torch.load("fine_tuned_cnn_efficientnet_b0.pth", map_location=torch.device('cpu'))  # Load efnet
resnet18 = models.resnet18(num_classes=10)
resnet18.load_state_dict(torch.load("fine_tuned_cnn_resnet18.pth", map_location=torch.device('cpu')))

#resnet_model = torch.load("fine_tuned_cnn_resnet18.pth", map_location=torch.device('cpu'))  # Load efnet
vit = models.vit_b_16(num_classes=10)
vit.load_state_dict(torch.load("fine_tuned_cnn_vit_b_16.pth", map_location=torch.device('cpu'))) # Load ViT

# Ensure CNN and ViT are in evaluation mode
rf_model
efnet.eval()
resnet18.eval()
vit.eval()

# Define CIFAR-10 class labels
class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", 
                "Dog", "Frog", "Horse", "Ship", "Truck"]

# Define preprocessing for CNN and ViT (224x224 for ViT, 32x32 for CIFAR)
transform_cnn = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize for CNN
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load an image (Replace 'test_image.png' with your own image)
image_path = "cat.jpg"
image = Image.open(image_path)

# Preprocess for ML models (SVM, RF) - Flattened 32x32x3 input
processed_image_ml = transform_cnn(image)
processed_image_ml = processed_image_ml.view(-1, 32*32*3)  # Flatten for SVM, RF
image_np = processed_image_ml.numpy()

# Preprocess for CNN (Tensor, no flattening)
processed_image_cnn = transform_cnn(image).unsqueeze(0)  # Add batch dimension

# Preprocess for ViT (224x224 input)
processed_image_vit = transform_vit(image).unsqueeze(0)  # Add batch dimension

# Predict using the trained models
rf_prediction = rf_model.predict(image_np)[0]
#svm_prediction = svm_model.predict(image_np)[0]

with torch.no_grad():
    efnet_output = efnet(processed_image_cnn)
    vit_output = vit(processed_image_vit)

efnet_prediction = torch.argmax(efnet_output, dim=1).item()
vit_prediction = torch.argmax(vit_output, dim=1).item()

# Display the image with predictions
plt.imshow(image)
plt.axis("off")
plt.title(f"RF: {class_labels[rf_prediction]} | \n"
          f"EfficientNet: {class_labels[efnet_prediction]} | ViT: {class_labels[vit_prediction]}")
plt.show()
