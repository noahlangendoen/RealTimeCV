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


    def preprocess_data(self, image_size=48):
        """
        Load and preprocess training and validation data

        Args:
            image_size: Target image size (default 48x48 for emotion recognition)
        """
        # Define data transforms with REDUCED augmentation for 48x48 images
        # FIXED: Previous augmentation was too aggressive for tiny faces
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),  # REDUCED: 15° -> 5° (prevents face distortion)
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # REDUCED: 10% -> 5%
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # REDUCED: 0.3 -> 0.1 (preserves expression cues)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # FIXED: Use simple normalization for faces
        ])

        # Validation transform without augmentation
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # FIXED: Match training normalization
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

            # FIXED: Remove normalization that causes extreme weight ratios
            # Use sqrt to moderate the effect of class imbalance
            class_weights = torch.sqrt(class_weights)

            # Print class distribution and weights for debugging
            print("\n" + "="*60)
            print("CLASS DISTRIBUTION:")
            for i, (class_name, count) in enumerate(zip(self.class_names, class_counts)):
                pct = 100 * count / total_samples
                print(f"  {class_name:15s}: {count:6d} samples ({pct:5.1f}%) | Weight: {class_weights[i]:.3f}")
            print("="*60 + "\n")

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using CrossEntropyLoss with class weights")

        if optimizer is None:
            # Use AdamW with weight decay for better generalization
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            print(f"Using AdamW optimizer with lr={self.learning_rate}, weight_decay=0.01")

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

            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_model48.pth')
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")

                # Also export to ONNX
                self._export_to_onnx(output_path="models/best_model48.onnx", image_size=48)

            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            # Check if model is stuck on one prediction
            unique_predictions = (class_predictions > 0).sum().item()
            if unique_predictions == 1:
                print(f"\n⚠ WARNING: Model is predicting only ONE class!")
                stuck_class = torch.argmax(class_predictions).item()
                print(f"  Stuck on class: {self.class_names[stuck_class]}")
                print(f"  This suggests the model has collapsed. Consider:")
                print(f"    - Reducing learning rate (current: {optimizer.param_groups[0]['lr']:.6f})")
                print(f"    - Checking class weights (they may be too extreme)")
                print(f"    - Restarting training with better initialization")
            elif unique_predictions <= 3:
                print(f"\n⚠ WARNING: Model is only predicting {unique_predictions} different classes!")

            # Show per-class accuracy and prediction distribution
            print("\nPer-class Performance:")
            for i, class_name in enumerate(self.class_names):
                if class_total[i] > 0:
                    class_acc = 100 * class_correct[i].item() / class_total[i].item()
                    pred_pct = 100 * class_predictions[i].item() / val_total
                    print(f"  {class_name:15s}: Acc={class_acc:5.1f}%  |  "
                          f"Predicted {class_predictions[i]:4d} times ({pred_pct:4.1f}%)")

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
        axes[0].set_title('Training and Validation Loss (48x48)')
        axes[0].legend()
        axes[0].grid(True)

        # Plot training and validation accuracy
        axes[1].plot(self.train_accuracies, label='Train Accuracy', marker='o')
        axes[1].plot(self.val_accuracies, label='Validation Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy (48x48)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('training_performance_48.png')
        print("Performance plot saved as 'training_performance_48.png'")
        plt.show()

        # Print final metrics
        print("\nFinal Metrics:")
        print(f"Best Training Accuracy: {max(self.train_accuracies):.2f}%")
        print(f"Best Validation Accuracy: {max(self.val_accuracies):.2f}%")
        print(f"Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {self.val_losses[-1]:.4f}")

    def _export_to_onnx(self, output_path="models/best_model48.onnx", image_size=48):
        """
        Export the current model to ONNX format

        Args:
            output_path: Path to save the ONNX model
            image_size: Input image size (48x48)
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
                opset_version=17,
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
    Main function to train 48x48 facial expression recognition model
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from models.improved_model_fr_cnn_48 import ImprovedNet48

    print("=" * 70)
    print("RealTimeCV - Facial Emotion Recognition Training (48x48)")
    print("=" * 70)

    # Initialize model
    print("\nInitializing ImprovedNet48 model...")
    model = ImprovedNet48(num_classes=7)

    # Print model info
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer with hyperparameters
    trainer = TrainModel(
        model=model,
        batch_size=64,  # Increased batch size for 48x48 (smaller images = more GPU memory available)
        learning_rate=0.001  # FIXED: Increased from 0.0001 to overcome initial saturation
    )

    # Preprocess and load data - IMPORTANT: Use 48x48 images!
    print("\nLoading and preprocessing data (48x48)...")
    trainer.preprocess_data(image_size=48)

    # Train the model
    print("\nStarting training...")
    trainer.train(epochs=30)

    # Analyze performance
    print("\nAnalyzing performance...")
    trainer.analyze_performance()

    print("\n" + "=" * 70)
    print("Training complete! Best model saved as 'best_model48.pth'")
    print("ONNX model exported to 'models/best_model48.onnx'")
    print("=" * 70)


if __name__ == "__main__":
    main()
