# train.py
# This script handles the training of our Vision Transformer model.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import json

# --- 1. Configuration ---
# Here we define all the settings for our training process.

# Directory where the dataset is stored.
DATA_DIR = "dataset"
# The specific pre-trained Vision Transformer model we'll use from Hugging Face.
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
# The number of disease categories (Early Blight, Late Blight, Healthy).
NUM_LABELS = 3
# Number of images to process in one go.
# NOTE: Reduced from 16 to 8 to fit within 4GB GPU VRAM.
# This is a crucial adjustment for GPUs with less memory.
BATCH_SIZE = 8
# How many times we'll go through the entire dataset.
NUM_EPOCHS = 10
# The learning rate for the optimizer.
LEARNING_RATE = 2e-5
# The device to train on (GPU if available, otherwise CPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Filepath to save our final trained model.
OUTPUT_MODEL_PATH = "potato_disease_model.pth"
# Filepath to save the accuracy plot.
ACCURACY_PLOT_PATH = "training_accuracy.png"

print(f"Using device: {DEVICE}")

# --- 2. Data Preparation ---
# We need to process our images into a format the model can understand.
# This includes resizing, converting to tensors, and data augmentation.

# Data augmentation for the training set to make the model more robust.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),      # ViT expects 224x224 images.
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally.
    transforms.RandomRotation(15),      # Randomly rotate images.
    transforms.ToTensor(),              # Convert image to a PyTorch tensor.
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize tensor values.
])

# For the validation set, we only resize, convert to tensor, and normalize.
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- 3. Load Datasets ---
print("Loading datasets...")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory not found at '{DATA_DIR}'. Please follow the README to set it up.")

# Create dataset objects from the image folders.
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

# Create data loaders to feed data to the model in batches.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Found {len(train_dataset)} training images belonging to {len(train_dataset.classes)} classes.")
print(f"Found {len(val_dataset)} validation images.")
print(f"Classes: {train_dataset.classes}")
print(f"Class to index mapping: {train_dataset.class_to_idx}")

# --- 4. Initialize Model, Optimizer, and Loss Function ---
print(f"Initializing Vision Transformer model: {MODEL_NAME}")
# Load the pre-trained ViT model and adapt its final layer for our 3 classes.
model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
model.to(DEVICE) # Move the model to the selected device (GPU/CPU).

# AdamW is an optimizer well-suited for transformer models.
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# CrossEntropyLoss is the standard loss function for multi-class classification.
criterion = nn.CrossEntropyLoss()

# --- 5. Training and Validation Loop ---
train_accuracies = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train() # Set the model to training mode.
    running_loss = 0.0
    train_preds, train_labels = [], []
    
    print(f"\n--- Starting Epoch {epoch+1}/{NUM_EPOCHS} ---")
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad() # Reset gradients.
        
        # Forward pass: get model predictions.
        outputs = model(images)
        # Calculate the loss.
        loss = criterion(outputs.logits, labels)
        # Backward pass: calculate gradients.
        loss.backward()
        # Update model weights.
        optimizer.step()

        running_loss += loss.item()
        
        preds = torch.argmax(outputs.logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        
        if (i+1) % 40 == 0: # Print progress update every 40 batches.
             print(f"  Batch {i+1}/{len(train_loader)}, Current Loss: {loss.item():.4f}")

    train_accuracy = accuracy_score(train_labels, train_preds)
    train_accuracies.append(train_accuracy)
    
    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode.
    val_preds, val_labels = [], []
    with torch.no_grad(): # Disable gradient calculation for validation.
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_accuracy)
    
    print(f"\nEpoch {epoch+1} Results:")
    print(f"  Average Training Loss: {running_loss / len(train_loader):.4f}")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")

# --- 6. Save the Final Model ---
# UPGRADED: Save a checkpoint dictionary for more robust loading.
print(f"\nTraining complete. Saving model checkpoint to '{OUTPUT_MODEL_PATH}'")
checkpoint = {
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': train_dataset.class_to_idx,
}
torch.save(checkpoint, OUTPUT_MODEL_PATH)
print("Checkpoint saved successfully.")


# --- 7. Plot and Save Accuracy Graph ---
plt.figure(figsize=(12, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, 'o-', label='Training Accuracy')
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, 'o-', label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(1, NUM_EPOCHS + 1))
plt.legend()
plt.grid(True)
plt.savefig(ACCURACY_PLOT_PATH)
print(f"Accuracy plot saved to '{ACCURACY_PLOT_PATH}'")

