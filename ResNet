# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 04:06:45 2024

@author: Administrator
"""

# ResNet50
# Ben PhD Scholar

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths (as specified by the user)
metadata_path = r'filtered_metadata.csv'  # Updated to the new filtered metadata file
device = torch.device('cpu')  # Force CPU for lightweight testing

# Custom Dataset Class
class SkinLesionDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        self.metadata = metadata_df
        self.transform = transform

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.metadata['encoded_label'] = self.label_encoder.fit_transform(self.metadata['label'])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = row['image_path']

        # Load and transform the image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(row['encoded_label'], dtype=torch.long)
        }

# Lightweight ResNet50 Model
class LightweightResNet50(nn.Module):
    def __init__(self, num_classes):
        super(LightweightResNet50, self).__init__()

        # Load pretrained ResNet50 with updated weights parameter
        weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)

        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze all layers for lightweight performance
        for param in self.features.parameters():
            param.requires_grad = False

        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 64),  # Further reduced complexity
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image):
        # Extract image features
        image_features = self.features(image)
        image_features = torch.flatten(image_features, 1)

        # Classify
        return self.classifier(image_features)

# Lightweight Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=2):
    print("Starting Lightweight Training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Main Execution
def main():
    # Load filtered metadata
    metadata = pd.read_csv(metadata_path)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Smaller image size for faster processing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset
    dataset = SkinLesionDataset(metadata, transform=transform)

    # Split dataset into training and validation
    train_metadata, val_metadata = train_test_split(
        metadata, test_size=0.1, stratify=metadata['label'], random_state=42  # Smaller split
    )

    # Create DataLoaders
    train_dataset = SkinLesionDataset(train_metadata, transform=transform)
    val_dataset = SkinLesionDataset(val_metadata, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Small batch size
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Initialize model, loss function, and optimizer
    num_classes = len(metadata['label'].unique())
    model = LightweightResNet50(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Faster learning rate

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer)

if __name__ == '__main__':
main()
