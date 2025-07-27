"""
Training script for ConforMorph-VAE.
Handles data loading, training loop, and model checkpointing.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
import time
import json
from tqdm import tqdm
import numpy as np

from models import ConforMorphVAE, vae_loss, count_parameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MolecularDataset(Dataset):
    """
    Dataset class for molecular coordinate data.
    """
    
    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to processed data file (.pt)
        """
        self.data = torch.load(data_path)
        self.coordinates = self.data['coordinates']
        self.masks = self.data['masks']
        
        logger.info(f"Loaded dataset: {len(self.coordinates)} molecules")
        logger.info(f"Coordinate shape: {self.coordinates.shape}")
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        return {
            'coordinates': self.coordinates[idx],
            'mask': self.masks[idx]
        }

def create_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create training and validation data loaders.
    
    Args:
        data_dir (str): Directory containing processed data
        batch_size (int): Batch size
        num_workers (int): Number of data loader workers
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    data_path = Path(data_dir)
    
    # Load datasets
    train_dataset = MolecularDataset(data_path / "train.pt")
    val_dataset = MolecularDataset(data_path / "val.pt")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    """
    Train for one epoch.
    
    Args:
        model: VAE model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        beta: KLD weighting factor
        
    Returns:
        dict: Training metrics
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        coordinates = batch['coordinates'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(coordinates, masks)
        
        # Compute loss
        loss_dict = vae_loss(
            outputs['recon_x'], coordinates,
            outputs['mu'], outputs['logvar'],
            masks, beta
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss_dict['total_loss'].item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_kld_loss += loss_dict['kld_loss'].item()
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kld_loss': total_kld_loss / num_batches
    }

def validate_epoch(model, val_loader, device, beta=1.0):
    """
    Validate for one epoch.
    
    Args:
        model: VAE model
        val_loader: Validation data loader
        device: Device to use
        beta: KLD weighting factor
        
    Returns:
        dict: Validation metrics
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            coordinates = batch['coordinates'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(coordinates, masks)
            
            # Compute loss
            loss_dict = vae_loss(
                outputs['recon_x'], coordinates,
                outputs['mu'], outputs['logvar'],
                masks, beta
            )
            
            # Accumulate metrics
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kld_loss += loss_dict['kld_loss'].item()
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kld_loss': total_kld_loss / num_batches
    }

def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, checkpoint_dir / "checkpoint_latest.pt")
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pt")

def main():
    parser = argparse.ArgumentParser(description="Train ConforMorph-VAE")
    
    # Data arguments
    parser.add_argument("--data", type=str, default="data/processed/", help="Data directory")
    
    # Model arguments
    parser.add_argument("--max_atoms", type=int, default=50, help="Maximum number of atoms")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--encoder_hidden", nargs='+', type=int, default=[512, 256, 128], 
                       help="Encoder hidden dimensions")
    parser.add_argument("--decoder_hidden", nargs='+', type=int, default=[128, 256, 512],
                       help="Decoder hidden dimensions")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="KLD weighting factor")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="experiments/", help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directories
    if args.run_name is None:
        args.run_name = f"run_{int(time.time())}"
    
    output_dir = Path(args.output_dir) / args.run_name
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data, args.batch_size, args.num_workers
    )
    
    # Initialize model
    model = ConforMorphVAE(
        max_atoms=args.max_atoms,
        latent_dim=args.latent_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.beta)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, args.beta)
        
        # Log metrics
        writer.add_scalar("Loss/Train_Total", train_metrics['total_loss'], epoch)
        writer.add_scalar("Loss/Train_Recon", train_metrics['recon_loss'], epoch)
        writer.add_scalar("Loss/Train_KLD", train_metrics['kld_loss'], epoch)
        writer.add_scalar("Loss/Val_Total", val_metrics['total_loss'], epoch)
        writer.add_scalar("Loss/Val_Recon", val_metrics['recon_loss'], epoch)
        writer.add_scalar("Loss/Val_KLD", val_metrics['kld_loss'], epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f} "
                   f"(Recon: {train_metrics['recon_loss']:.4f}, "
                   f"KLD: {train_metrics['kld_loss']:.4f})")
        logger.info(f"Val Loss: {val_metrics['total_loss']:.4f} "
                   f"(Recon: {val_metrics['recon_loss']:.4f}, "
                   f"KLD: {val_metrics['kld_loss']:.4f})")
        
        # Save checkpoint
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        save_checkpoint(
            model, optimizer, epoch,
            {'train': train_metrics, 'val': val_metrics},
            checkpoint_dir, is_best
        )
    
    # Training complete
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main()