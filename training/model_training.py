from dotenv import load_dotenv
import os

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import onnxscript

load_dotenv()


class TrainModel():
    def __init__(self, model, batch_size=32, learning_rate=0.0001, device=None):
        self.model = model
        self.train_dir = os.getenv("TRAIN_DIR")
        self.test_dir = os.getenv("TEST_DIR")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []


    def preprocess_data(self, image_size=224):
        """
        Load and preprocess training and validation data

        Args:
            image_size: Target image size (default 48x48 for emotion recognition)
        """
        # Define data transforms with enhanced augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])

        # Validation transform without augmentation
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load datasets using ImageFolder
        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=self.test_dir, transform=val_transform)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Store class names and counts
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)

        return self.train_loader, self.val_loader

    def train(self, epochs=10, criterion=None, optimizer=None):
        """
        Train the model

        Args:
            epochs: Number of training epochs
            criterion: Loss function (default: CrossEntropyLoss with class weights)
            optimizer: Optimizer (default: AdamW)
        """
        # Calculate class weights for imbalanced dataset
        if criterion is None:
            # Count samples per class
            class_counts = []
            for class_idx in range(len(self.train_loader.dataset.classes)):
                class_dir = os.path.join(self.train_dir, self.train_loader.dataset.classes[class_idx])
                class_counts.append(len(os.listdir(class_dir)))

            # Calculate weights (inverse of frequency)
            total_samples = sum(class_counts)
            class_weights = [total_samples / count for count in class_counts]
            class_weights = torch.FloatTensor(class_weights).to(self.device)

            # Normalize weights
            class_weights = class_weights / class_weights.sum() * len(class_weights)

            criterion = nn.CrossEntropyLoss(weight=class_weights)

        if optimizer is None:
            # Use AdamW with weight decay for better generalization
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )

        # Use ReduceLROnPlateau for adaptive learning rate reduction
        # This will reduce LR when validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3,
            min_lr=1e-6
        )

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Print progress
                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], "
                          f"Loss: {loss.item():.4f}")

            # Calculate training metrics
            avg_train_loss = train_loss / len(self.train_loader)
            train_accuracy = 100 * train_correct / train_total
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            # Track per-class predictions to detect if model is stuck on one class
            class_predictions = torch.zeros(self.num_classes, dtype=torch.long)
            class_correct = torch.zeros(self.num_classes, dtype=torch.long)
            class_total = torch.zeros(self.num_classes, dtype=torch.long)

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    # Track per-class statistics
                    for i in range(len(labels)):
                        label = labels[i].item()
                        pred = predicted[i].item()
                        class_predictions[pred] += 1
                        class_total[label] += 1
                        if label == pred:
                            class_correct[label] += 1

            # Calculate validation metrics
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = 100 * val_correct / val_total
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)

            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")

                # Also export to ONNX for C++ deployment
                self._export_to_onnx()

            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        print("Training completed!")
        return self.model


    def _export_to_onnx(self, output_path="models/best_model.onnx", image_size=224):
        """
        Export the current model to ONNX format
        
        Args:
            output_path: Path to save the ONNX model
            image_size: Input image size (default 48x48)
        """
        try:
            self.model.eval()
            dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=17,  # Changed from 11 to 17 (ONNX Runtime supports this well)
                do_constant_folding=True,  # Optimize the model
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                dynamo=False
            )
            
            print(f"✓ Model exported to ONNX: {output_path}")
            print(f"  Input: [batch_size, 3, {image_size}, {image_size}]")
            print(f"  Output: [batch_size, 7]")
            print(f"  Classes: {self.class_names if hasattr(self, 'class_names') else 'Unknown'}")
            
        except Exception as e:
            print(f"✗ Failed to export ONNX: {e}")
            import traceback
            traceback.print_exc()



def main():
    """
    Main function to test training pipeline with facial_recog_cnn model
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from models.facial_recog_cnn import EvenBetterNet
    from models.improved_model_fr_cnn import ImprovedNet

    print("=" * 70)
    print("RealTimeCV - Facial Emotion Recognition Training")
    print("=" * 70)

    # Initialize model
    print("\nInitializing EvenBetterNet model...")
    model = ImprovedNet()

    # Create trainer with hyperparameters
    trainer = TrainModel(
        model=model,
        batch_size=32,
        learning_rate=0.0001  # Reduced from 0.001 to prevent model collapse
    )

    # Preprocess and load data
    print("\nLoading and preprocessing data...")
    trainer.preprocess_data(image_size=224)

    # Train the model
    print("\nStarting training...")
    trainer.train(epochs=30)

    print("Training complete! Best model saved as 'best_model.pth'")


if __name__ == "__main__":
    main()