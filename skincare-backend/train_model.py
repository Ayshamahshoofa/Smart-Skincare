import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim

def main():
    # --------------------------
    # 1️⃣ Paths
    # --------------------------
    data_dir = r"C:\Users\AYISHATHAYISHATH\OneDrive\Documents\Desktop\smart-skincare\skincare-backend\data"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # --------------------------
    # 2️⃣ Transforms
    # --------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --------------------------
    # 3️⃣ Datasets
    # --------------------------
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"✅ Training images: {len(train_dataset)}")
    print(f"✅ Test images: {len(test_dataset)}")
    print(f"✅ Number of classes: {len(train_dataset.classes)}")

    # --------------------------
    # 4️⃣ Model setup (ResNet18)
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)

    # --------------------------
    # 5️⃣ Loss + Optimizer
    # --------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --------------------------
    # 6️⃣ Training Loop
    # --------------------------
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print(f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}%")

    # --------------------------
    # 7️⃣ Save the Model
    # --------------------------
    torch.save(model.state_dict(), "skincare_model.pth")
    print("\n✅ Model saved as 'skincare_model.pth'")

# ✅ Required for Windows multiprocessing
if __name__ == "__main__":
    main()
