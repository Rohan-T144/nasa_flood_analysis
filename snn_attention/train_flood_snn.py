#!/usr/bin/env python3
"""
Trains a Spiking U-Net with self-attention for flood mapping using PyTorch and SpikingJelly.
Based on the Spike2Former architecture, this script implements a spiking neural network (SNN) for semantic segmentation tasks.
Spike2Former: https://github.com/BICLab/Spike2Former/tree/main
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import torch.optim as optim
from model_snn import SpikingUNet  # Import the SNN model
from skimage import io
from spikingjelly.activation_based import functional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

if torch.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
torch.multiprocessing.set_sharing_strategy('file_system')

# --- Dataset Class (copied from your original script) ---
class FloodDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = io.imread(img_path).astype(np.float32)

        # Normalize each band to [0, 1]
        for band in range(image.shape[2]):
            band_data = image[:, :, band]
            min_val = np.min(band_data)
            max_val = np.max(band_data)
            if max_val > min_val:
                image[:, :, band] = (band_data - min_val) / (max_val - min_val)
            else:
                image[:, :, band] = 0
        
        # Transpose from [H, W, C] to [C, H, W]
        image = np.transpose(image, (2, 0, 1))

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.img_files[idx].replace('.tif', '.png'))
            if os.path.exists(mask_path):
                mask = io.imread(mask_path).astype(np.float32)
                mask = mask / 255.0  # Normalize to [0, 1]
                mask = mask[np.newaxis, :, :]  # Add channel dim [1, H, W]
            else:
                mask = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.float32)
        else:
            mask = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.float32)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return torch.from_numpy(image), torch.from_numpy(mask)

# --- Loss and Metrics (copied from your original script) ---
def dice_coefficient(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy(pred, target)
    dice = 1 - dice_coefficient(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice

# --- Main Training and Testing Functions ---
def train_model(args):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # --- SNN Model Instantiation ---
    model = SpikingUNet(n_channels=12, n_classes=1, bilinear=True, T=args.timesteps)
    model.to(device)

    # Initialize training state
    start_epoch = 0
    best_loss = float('inf')
    train_history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Load checkpoint if it exists
    # checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")   
    checkpoint_path = os.path.join(args.save_dir, "snn_unet_best.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        train_history = checkpoint['train_history']
        print(f"Resuming from epoch {start_epoch}, best validation loss: {best_loss:.4f}")
    
    # --- DataLoaders ---
    train_dataset_full = FloodDataset(
        img_dir=os.path.join(args.data_dir, 'train', 'images'),
        mask_dir=os.path.join(args.data_dir, 'train', 'labels')
    )
    
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_full, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        start_time = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            
            # --- SNN Forward Pass ---
            # The model's forward pass handles the timesteps internally
            outputs = model(images)
            
            loss = bce_dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # --- SNN Reset ---
            # Reset the state of the network (membrane potential, etc.)
            functional.reset_net(model)

            dice = dice_coefficient(outputs, masks)
            train_loss += loss.item()
            train_dice += dice.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        epoch_time = time.time() - start_time

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                loss = bce_dice_loss(outputs, masks)
                dice = dice_coefficient(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
                
                # Reset is important in validation too if the model has state
                functional.reset_net(model)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step(val_loss)

        # Update training history
        train_history['train_loss'].append(train_loss)
        train_history['train_dice'].append(train_dice)
        train_history['val_loss'].append(val_loss)
        train_history['val_dice'].append(val_dice)

        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'train_history': train_history
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(args.save_dir, "snn_unet_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        
        # Save periodic model checkpoint
        if (epoch + 1) % args.save_every == 0:
            periodic_path = os.path.join(args.save_dir, f"snn_unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), periodic_path)
            print(f"Periodic model saved to {periodic_path}")
        
        print("-" * 80)

    print("Training complete!")
    return model

def test_model(model, args):
    test_dataset = FloodDataset(
        img_dir=os.path.join(args.data_dir, 'train', 'images'),
        mask_dir=os.path.join(args.data_dir, 'train', 'labels'),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            if idx >= args.num_visualizations:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            binary_preds = (outputs > 0.5).float()
            
            # Reset after each inference
            functional.reset_net(model)

            # Visualization logic (same as your original script)
            image = images.cpu().numpy()[0]
            pred = outputs.cpu().numpy()[0, 0]
            mask = masks.cpu().numpy()[0, 0]
            binary_pred = binary_preds.cpu().numpy()[0, 0]
            
            rgb_image = np.stack([image[3], image[2], image[1]], axis=2) # False color
            rgb_image = np.clip(rgb_image, 0, 1)
            
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(rgb_image)
            axes[0].set_title("False Color Image (B4,B3,B2)")
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='Blues')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            axes[2].imshow(pred, cmap='plasma', vmin=0, vmax=1)
            axes[2].set_title("SNN Prediction (Probability)")
            axes[2].axis('off')

            axes[3].imshow(binary_pred, cmap='Blues')
            axes[3].set_title("SNN Binary Prediction")
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"snn_prediction_{idx}.png"))
            plt.close()

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(test_loader)} test images")
    
    print(f"Testing complete! Predictions saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spiking U-Net for flood mapping")
    parser.add_argument("--data-dir", type=str, default="../flood_data", help="Path to data directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training. SNNs are memory intensive.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--timesteps", "-T", type=int, default=4, help="Number of timesteps for SNN simulation")
    parser.add_argument("--save-dir", type=str, default="snn_checkpoints", help="Directory to save models")
    parser.add_argument("--output-dir", type=str, default="snn_predictions", help="Directory to save predictions")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--num-visualizations", type=int, default=10, help="Number of predictions to visualize")
    parser.add_argument("--test-only", action="store_true", help="Only run testing, no training")
    parser.add_argument("--model-path", type=str, default="snn_checkpoints/snn_unet_best.pth", help="Path to model for testing")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--save-every", type=int, default=5, help="Save model every N epochs")

    args = parser.parse_args()
    
    if torch.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    if args.test_only:
        if not os.path.exists(args.model_path):
            print(f"Error: Model path not found at {args.model_path}")
        else:
            model = SpikingUNet(n_channels=12, n_classes=1, T=args.timesteps)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.to(device)
            print(f"Loaded model from {args.model_path}")
            test_model(model, args)
    else:
        model = train_model(args)
        test_model(model, args)
