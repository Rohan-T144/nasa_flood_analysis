#!/usr/bin/env python3
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model_ann import ANNUNet
from skimage import io
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


# Dataset class for flood data
class FloodDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get all image files
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')], key=lambda name : int(name[:-4]))
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        
        # Read image
        image = io.imread(img_path).astype(np.float32)

        # Normalize image (assuming image is 12-band)
        for band in range(image.shape[2]):
            band_data = image[:, :, band]
            min_val = np.min(band_data)
            max_val = np.max(band_data)
            if max_val > min_val:
                image[:, :, band] = (band_data - min_val) / (max_val - min_val)
            else:
                image[:, :, band] = 0

        # Transpose image from [H, W, C] to [C, H, W]
        image = np.transpose(image, (2, 0, 1))

        # Load mask (if available)
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.img_files[idx].replace('.tif', '.png'))
            if os.path.exists(mask_path):
                mask = io.imread(mask_path).astype(np.float32)
                mask = (mask > 0).astype(np.float32)
                mask = mask[np.newaxis, :, :]  # Add channel dimension [1, H, W]
            else:
                # Create dummy mask for test images
                mask = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.float32)
        else:
            # Create dummy mask for test images
            mask = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.float32)

        # Apply any transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return torch.from_numpy(image), torch.from_numpy(mask)

# Dice coefficient for evaluation
def dice_coefficient(pred, target, smooth=1.0):
    """Calculate Dice coefficient between predicted and target masks"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()

    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# Binary cross-entropy dice loss
def bce_dice_loss(pred, target, bce_weight=0.5):
    """Combined BCE and Dice loss for better segmentation results"""
    bce = F.binary_cross_entropy(pred, target)

    dice = dice_coefficient(pred, target)

    return bce_weight * bce + (1 - bce_weight) * (1 - dice)

def train_model(args):
    # Set device
    if torch.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create model
    model = ANNUNet(n_channels=12, n_classes=1, bilinear=True)

    # Load checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")
    else:
        start_epoch = 0
        best_loss = float('inf')

    model = model.to(device)

    # # Create train dataset and dataloader
    # train_dataset = FloodDataset(
    #     img_dir=os.path.join(args.data_dir, 'train', 'images'),
    #     mask_dir=os.path.join(args.data_dir, 'train', 'labels')
    # )
    
    # # Create validation dataset and dataloader
    # val_dataset = FloodDataset(
    #     img_dir=os.path.join(args.data_dir, 'val', 'images'),
    #     # Note: validation set has no labels, so we'll use random split from train for validation
    # )
    
    # # Split training data into train and validation sets if no validation labels
    # if not os.path.exists(os.path.join(args.data_dir, 'val', 'labels')):
    #     print("No validation labels found. Using split from training data.")
    #     train_size = int(0.8 * len(train_dataset))
    #     val_size = len(train_dataset) - train_size
    #     train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # # Create dataloaders
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- DataLoaders ---
    train_dataset_full = FloodDataset(
        img_dir=os.path.join(args.data_dir, 'train', 'images'),
        mask_dir=os.path.join(args.data_dir, 'train', 'labels')
    )
    
    
    # train_size = int(0.8 * len(train_dataset_full))
    # val_size = len(train_dataset_full) - train_size
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    if args.data_seed != None:
        print(f"Generating data split with seed {args.data_seed}...")
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset_full, 
            [train_size, val_size, test_size],
            generator = torch.Generator().manual_seed(args.data_seed),
        )
    else:
        print("Generating random data split...")
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset_full, 
            [train_size, val_size, test_size],
        )
    
    os.makedirs("dataloaders", exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    torch.save(test_loader, "dataloaders/testloader.pt")

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Load optimizer state if resuming training
    if args.resume and os.path.exists(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded optimizer and scheduler states")

    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        start_time = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = bce_dice_loss(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate dice coefficient
            dice = dice_coefficient(outputs, masks)

            # Update metrics
            train_loss += loss.item()
            train_dice += dice.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")

        # Calculate average metrics for the epoch
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        epoch_time = time.time() - start_time

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate loss and dice
                loss = bce_dice_loss(outputs, masks)
                dice = dice_coefficient(outputs, masks)

                # Update metrics
                val_loss += loss.item()
                val_dice += dice.item()

        # Calculate average metrics for validation
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Update scheduler
        scheduler.step(val_loss)

        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # Save model if it's the best so far
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(args.save_dir, "ann_unet_best.pth")
            # Save model with additional information for resuming training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice
            }, save_path)
            print(f"Model saved to {save_path}")

        # Save model at regular intervals
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"ann_unet_epoch_{epoch+1}.pth")
            # Save model with additional information for resuming training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice
            }, save_path)
            print(f"Model saved to {save_path}")

        print("-" * 80)

    print("Training complete!")
    return model

def test_model(model, args):
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # # Create dataset
    # test_dataset = FloodDataset(
    #     img_dir=os.path.join(args.data_dir, 'train', 'images'),
    #     mask_dir=os.path.join(args.data_dir, 'train', 'labels'),
    # )

    # # Create dataloader
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_loader = torch.load("dataloaders/testloader.pt", weights_only=False)

    # Set model to evaluation mode
    model.eval()

    # Create directory for saving predictions
    os.makedirs(args.output_dir, exist_ok=True)

    test_loss = 0.0
    test_dice = 0.0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            # Forward pass
            outputs = model(images)

            # Create binary prediction
            #binary_preds = (outputs > 0.01).float()
            binary_preds = (outputs > 0.5).float()
            # print(outputs.min(), outputs.max(), binary_preds.min(), binary_preds.max())

            loss = bce_dice_loss(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            test_loss += loss.item()
            test_dice += dice.item()

            for i in range(len(images)):
                # Save prediction visualization
                if idx*args.batch_size + i < args.num_visualizations:
                    # Convert tensors to numpy arrays
                    image = images.cpu().numpy()[0]
                    pred = outputs.cpu().numpy()[0, 0]
                    mask = masks.cpu().numpy()[0, 0]
                    binary_pred = binary_preds.cpu().numpy()[0, 0]

                    # Create RGB visualization using bands 0, 1, 2
                    rgb_image = np.stack([
                        image[0],  # Red channel (band 0)
                        image[1],  # Green channel (band 1)
                        image[2]   # Blue channel (band 2)
                    ], axis=2)

                    # Clip values to [0, 1] range
                    rgb_image = np.clip(rgb_image, 0, 1)

                    # Create figure with 3 subplots
                    fig, axes = plt.subplots(2, 2, figsize=(15, 5))

                    # Plot RGB image
                    axes[0, 0].imshow(rgb_image)
                    axes[0, 0].set_title("RGB Image (Bands 0,1,2)")
                    axes[0, 0].axis('off')
                    
                    # Plot ground truth mask
                    axes[0, 1].imshow(mask, cmap='Blues')
                    axes[0, 1].set_title("Ground Truth")
                    axes[0, 1].axis('off')

                    # Plot probability prediction
                    axes[1, 0].imshow(pred, cmap='plasma')
                    axes[1, 0].set_title("Prediction (Probability)")
                    axes[1, 0].axis('off')

                    # Plot binary prediction
                    axes[1, 1].imshow(binary_pred, cmap='Blues')
                    axes[1, 1].set_title("Binary Prediction")
                    axes[1, 1].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, f"prediction_{idx}.png"))
                    plt.close()

                    # Save the binary prediction as a mask
                    # binary_mask = (binary_pred * 255).astype(np.uint8)
                    # io.imsave(os.path.join(args.output_dir, f"mask_{idx}.png"), binary_mask)

                if (idx*args.batch_size + i) % 10 == 0:
                    print(f"Processed {idx*args.batch_size + i}/{len(test_loader) * args.batch_size} test images")
    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")

    print(f"Testing complete! Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for flood mapping")
    parser.add_argument("--data-dir", type=str, default="../flood_data", help="Path to data directory")
    parser.add_argument("--data-seed", type=int, help="Seed to generate a random train/val/test split, random if not set")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="ann_checkpoints", help="Directory to save models")
    parser.add_argument("--save-interval", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    parser.add_argument("--output-dir", type=str, default="ann_predictions", help="Directory to save predictions")
    parser.add_argument("--num-visualizations", type=int, default=10, help="Number of predictions to visualize")
    parser.add_argument("--test-only", action="store_true", help="Only run testing, no training")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model for testing")
    

    args = parser.parse_args()

    if args.test_only and args.model_path:
        # Load model for testing
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = ANNUNet(n_channels=12, n_classes=1)

        # Load checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        # Handle both new checkpoint format and old format (model_state_dict only)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch {checkpoint['epoch']})")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {args.model_path}")

        model = model.to(device)
        test_model(model, args)
    else:
        # Train model
        model = train_model(args)
        # Test the trained model
        test_model(model, args)