import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

# Load your model
num_classes = 31
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load("skincare_model.pth", map_location="cpu"))
model.eval()

# Define the same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Test on one image
img_path = r"C:\Users\AYISHATHAYISHATH\OneDrive\Documents\Desktop\smart-skincare\skincare-backend\data\test\Hailey-Hailey Disease\2381.jpg"
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(img)
    _, pred = outputs.max(1)
    print(f"Predicted class index: {pred.item()}")
