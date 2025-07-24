#!/usr/bin/env python3
"""
Trains a Spiking U-Net with self-attention for flood mapping using PyTorch and SpikingJelly.
"""
import argparse
import os
import time
import json

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
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')], key=lambda name : int(name[:-4]))

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
                # ensure values strictly 0 or 1
                mask = (mask > 0).astype(np.float32)
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

def calculate_iou(pred, target, threshold=0.5):
    """
    Calculates IoU for binary segmentation.
    """
    # Apply sigmoid and threshold to get binary predictions
    # pred_prob = torch.sigmoid(pred_logits)
    pred_mask = (pred > threshold) # Convert to 0s and 1s

    pred_mask = pred_mask.view(-1)
    target = target.view(-1)

    pred_mask = pred_mask.to(torch.uint8)
    target = target.to(torch.uint8)

    # For binary case (num_classes=1), we calculate IoU for the positive class (class 1)
    # This assumes the target mask is also 0s and 1s.
    intersection = (pred_mask & target).sum().item()
    union = (pred_mask | target).sum().item()
    
    if union == 0:
        # If there is no ground truth or prediction for the positive class,
        # it can be considered a perfect match (IoU=1) or nan.
        # Returning nan is safer to indicate the case.
        iou = float('nan')
    else:
        iou = intersection / union
        
    # Since it's binary, we return the single IoU score.
    return torch.tensor(iou)

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
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")   
    # checkpoint_path = os.path.join(args.save_dir, "snn_unet_best.pth")
    # checkpoint_path = os.path.join(args.save_dir, "spike_unet_former_checkpoint.pth")
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
        
    os.makedirs(args.save_dir, exist_ok=True)
    #os.makedirs("dataloaders", exist_ok=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    torch.save(test_loader, os.path.join(args.save_dir, "testloader.pt"))

    
    # THIS IS FOR ADDING POS_WEIGHT TO BCE LOSS, NOT YET IMPLEMENTED
    # WOULD NEED TO CHANGE MODEL TO OUTPUT LOGITS AND USE BCEWITHLOGITS

    # # Add this before your training loop
    # num_non_flood_pixels = 0
    # num_flood_pixels = 0
    # # Use your training data loader to iterate through the masks
    # for _, mask in train_loader:
    #     num_non_flood_pixels += (mask == 0).sum().item()
    #     num_flood_pixels += (mask == 1).sum().item()
    # print(num_non_flood_pixels, num_flood_pixels)
    # # The weight is the ratio of negatives to positives
    # pos_weight = num_non_flood_pixels / num_flood_pixels
    # print(f"Calculated positive class weight: {pos_weight:.2f}")

    # # Convert it to a tensor for the loss function
    # pos_weight_tensor = torch.tensor([pos_weight], device=device)


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

            # if (batch_idx + 1) % 10 == 0:
            #     outputs, spike_outputs = model(images, return_spikes=True)

            #     print("--- Layer-wise Spike Rate Analysis ---")
            #     def print_spike_rate(tensor, name="Layer"):
            #         # tensor shape: [T, Batch, Channels, H, W]
            #         num_spikes = torch.sum(tensor > 0).item()
            #         num_neurons = tensor.numel() / tensor.shape[0] # Total neurons in the batch
            #         avg_spikes_per_neuron = num_spikes / num_neurons
                    
            #         # Firing rate as a percentage of timesteps
            #         firing_rate_percent = (avg_spikes_per_neuron / tensor.shape[0]) * 100
                    
            #         print(f"{name}: Avg Spikes/Neuron = {avg_spikes_per_neuron:.4f} | Firing Rate = {firing_rate_percent:.2f}%")
            #     for name, spikes in spike_outputs.items():
            #         print_spike_rate(spikes.detach(), name=name)
            # else:
            #     outputs = model(images)
            
            loss = bce_dice_loss(outputs, masks)
            loss.backward()

            # # --- Gradient Check ---
            # total_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # if (batch_idx + 1) % 10 == 0:
            #     print(f"Gradient Norm: {total_norm:.4f}")
            # # --- End Gradient Check ---

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
    # test_dataset = FloodDataset(
    #     img_dir=os.path.join(args.data_dir, 'train', 'images'),
    #     mask_dir=os.path.join(args.data_dir, 'train', 'labels'),
    # )
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    save_dir = os.path.dirname(args.model_path)
    test_loader = torch.load(os.path.join(save_dir, "testloader.pt"), weights_only=False)

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            # if idx >= args.num_visualizations:
            #     break
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            binary_preds = (outputs > 0.5).float()
            
            # Reset after each inference
            functional.reset_net(model)

            loss = bce_dice_loss(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            iou = calculate_iou(outputs, masks)
            test_loss += loss.item()
            test_dice += dice.item()
            test_iou += iou.item()

            # if idx*args.batch_size + i < args.num_visualizations:
            for i in range(len(images)):
                if idx*args.batch_size + i < args.num_visualizations:
                    image = images.cpu().numpy()[i]
                    pred = outputs.cpu().numpy()[i, 0]
                    mask = masks.cpu().numpy()[i, 0]
                    binary_pred = binary_preds.cpu().numpy()[i, 0]
                    
                    rgb_image = np.stack([image[3], image[2], image[1]], axis=2) # False color
                    rgb_image = np.clip(rgb_image, 0, 1)
                    
                    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
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
                    plt.savefig(os.path.join(args.output_dir, f"snn_prediction_{idx*args.batch_size + i}.png"))
                    plt.close()

                if (idx*args.batch_size + i) % 10 == 0:
                    print(f"Processed {idx*args.batch_size + i}/{len(test_loader) * args.batch_size} test images")
    
    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    test_iou /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}, Test IoU: {test_iou:.4f}")
    
    metrics = {"test loss" : test_loss, "test dice" : test_dice, "test_iou" : test_iou}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as json_file:
        json.dump(metrics, json_file, indent=4)
    
    print(f"Testing complete! Predictions saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spiking U-Net for flood mapping")
    parser.add_argument("--data-dir", type=str, default="../flood_data", help="Path to data directory")
    parser.add_argument("--data-seed", type=int, help="Seed to generate a random train/val/test split, random if not set")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training. SNNs are memory intensive.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--timesteps", "-T", type=int, default=4, help="Number of timesteps for SNN simulation")
    parser.add_argument("--save-dir", type=str, default="snn_checkpoints", help="Directory to save models")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--save-every", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    
    parser.add_argument("--output-dir", type=str, default="snn_predictions", help="Directory to save predictions")
    parser.add_argument("--num-visualizations", type=int, default=10, help="Number of predictions to visualize")
    parser.add_argument("--test-only", action="store_true", help="Only run testing, no training")
    parser.add_argument("--model-path", type=str, help="Path to model for testing")
    
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"{args.save_dir}/snn_unet_best.pth"
    
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
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            print(f"Loaded model from {args.model_path}")
            test_model(model, args)
    else:
        # save args
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
        model = train_model(args)
        test_model(model, args)
