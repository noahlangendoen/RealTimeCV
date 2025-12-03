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

load_dotenv()

class TrainModel():
    def __init__(self, model, batch_size=32, learning_rate=0.001, device=None):
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


    def preprocess_data(self, image_size=48):
        """
        Load and preprocess training and validation data

        Args:
            image_size: Target image size (default 48x48 for emotion recognition)
        """
        # Define data transforms with augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
        print(f"Classes: {self.class_names}")
        print(f"Device: {self.device}")

        return self.train_loader, self.val_loader

    def train(self, epochs=10, criterion=None, optimizer=None):
        """
        Train the model

        Args:
            epochs: Number of training epochs
            criterion: Loss function (default: CrossEntropyLoss with class weights)
            optimizer: Optimizer (default: Adam)
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

            criterion = nn.CrossEntropyLoss(weight=class_weights)

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
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

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate validation metrics
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = 100 * val_correct / val_total
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")

            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            print("-" * 60)

        print("Training completed!")
        return self.model

    def analyze_performance(self):
        """
        Analyze and visualize model performance
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training and validation loss
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot training and validation accuracy
        axes[1].plot(self.train_accuracies, label='Train Accuracy', marker='o')
        axes[1].plot(self.val_accuracies, label='Validation Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('training_performance.png')
        print("Performance plot saved as 'training_performance.png'")
        plt.show()

        # Print final metrics
        print("\nFinal Metrics:")
        print(f"Best Training Accuracy: {max(self.train_accuracies):.2f}%")
        print(f"Best Validation Accuracy: {max(self.val_accuracies):.2f}%")
        print(f"Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {self.val_losses[-1]:.4f}")


def main():
    """
    Main function to test training pipeline with facial_recog_cnn model
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from models.facial_recog_cnn import EvenBetterNet

    print("=" * 70)
    print("RealTimeCV - Facial Emotion Recognition Training")
    print("=" * 70)

    # Initialize model
    print("\nInitializing EvenBetterNet model...")
    model = EvenBetterNet()

    # Create trainer with hyperparameters
    trainer = TrainModel(
        model=model,
        batch_size=32,
        learning_rate=0.001
    )

    # Preprocess and load data
    print("\nLoading and preprocessing data...")
    trainer.preprocess_data(image_size=48)

    # Train the model
    print("\nStarting training...")
    trainer.train(epochs=30)

    # Analyze performance
    print("\nAnalyzing performance...")
    trainer.analyze_performance()

    print("\n" + "=" * 70)
    print("Training complete! Best model saved as 'best_model.pth'")
    print("=" * 70)


if __name__ == "__main__":
    main()