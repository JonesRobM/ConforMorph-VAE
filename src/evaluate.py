"""
Evaluation script for ConforMorph-VAE.
Computes reconstruction metrics and performs latent space analysis.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models import ConforMorphVAE, vae_loss
from train import MolecularDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_rmsd(coords1, coords2, mask=None):
    """
    Compute Root Mean Square Deviation between two coordinate sets.
    
    Args:
        coords1, coords2 (torch.Tensor): Coordinate tensors [batch_size, max_atoms, 3]
        mask (torch.Tensor): Atom mask [batch_size, max_atoms]
        
    Returns:
        torch.Tensor: RMSD values for each molecule [batch_size]
    """
    # Compute squared differences
    squared_diff = (coords1 - coords2) ** 2
    
    if mask is not None:
        # Apply mask and compute mean over valid atoms
        mask_expanded = mask.unsqueeze(-1)  # [batch_size, max_atoms, 1]
        masked_diff = squared_diff * mask_expanded
        
        # Sum over atoms and coordinates, divide by number of valid atoms
        sum_diff = masked_diff.sum(dim=[1, 2])  # [batch_size]
        num_atoms = mask.sum(dim=1) * 3  # [batch_size] (3 coordinates per atom)
        rmsd = torch.sqrt(sum_diff / num_atoms)
    else:
        # Simple RMSD without masking
        rmsd = torch.sqrt(squared_diff.mean(dim=[1, 2]))
    
    return rmsd

def evaluate_reconstruction(model, data_loader, device):
    """
    Evaluate reconstruction quality on a dataset.
    
    Args:
        model: Trained VAE model
        data_loader: Data loader for evaluation
        device: Device to use
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_rmsd = []
    all_recon_loss = []
    all_kld_loss = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating reconstruction"):
            coordinates = batch['coordinates'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(coordinates, masks)
            
            # Compute RMSD
            rmsd = compute_rmsd(outputs['recon_x'], coordinates, masks)
            all_rmsd.extend(rmsd.cpu().numpy())
            
            # Compute losses
            loss_dict = vae_loss(
                outputs['recon_x'], coordinates,
                outputs['mu'], outputs['logvar'],
                masks
            )
            
            all_recon_loss.extend([loss_dict['recon_loss'].item()] * len(coordinates))
            all_kld_loss.extend([loss_dict['kld_loss'].item()] * len(coordinates))
    
    all_rmsd = np.array(all_rmsd)
    
    return {
        'rmsd_mean': all_rmsd.mean(),
        'rmsd_std': all_rmsd.std(),
        'rmsd_median': np.median(all_rmsd),
        'rmsd_q75': np.percentile(all_rmsd, 75),
        'rmsd_q95': np.percentile(all_rmsd, 95),
        'rmsd_all': all_rmsd,
        'recon_loss_mean': np.mean(all_recon_loss),
        'kld_loss_mean': np.mean(all_kld_loss)
    }

def latent_space_analysis(model, data_loader, device, num_samples=1000):
    """
    Analyze the learned latent space.
    
    Args:
        model: Trained VAE model
        data_loader: Data loader
        device: Device to use
        num_samples: Number of samples to analyze
        
    Returns:
        dict: Latent space analysis results
    """
    model.eval()
    
    latent_vectors = []
    molecular_features = []
    
    with torch.no_grad():
        samples_collected = 0
        for batch in data_loader:
            if samples_collected >= num_samples:
                break
                
            coordinates = batch['coordinates'].to(device)
            masks = batch['mask'].to(device)
            
            # Encode to latent space
            mu, logvar = model.encoder(coordinates, masks)
            latent_vectors.append(mu.cpu())
            
            # Compute basic features
            num_atoms = masks.sum(dim=1).cpu().numpy()
            molecular_features.extend(num_atoms)
            
            samples_collected += len(coordinates)
    
    latent_vectors = torch.cat(latent_vectors, dim=0)[:num_samples]
    molecular_features = np.array(molecular_features)[:num_samples]
    
    # PCA analysis
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors.numpy())
    
    # t-SNE analysis (if enough samples)
    latent_tsne = None
    if len(latent_vectors) >= 50:
        tsne = TSNE(n_components=2, random_state=42)
        latent_tsne = tsne.fit_transform(latent_vectors.numpy())
    
    return {
        'latent_vectors': latent_vectors,
        'molecular_features': molecular_features,
        'pca': latent_pca,
        'tsne': latent_tsne,
        'pca_explained_variance': pca.explained_variance_ratio_
    }

def interpolate_molecules(model, dataset, device, mol_idx1=0, mol_idx2=1, num_steps=10):
    """
    Interpolate between two molecules in latent space.
    
    Args:
        model: Trained VAE model
        dataset: Dataset containing molecules
        device: Device to use
        mol_idx1, mol_idx2: Indices of molecules to interpolate between
        num_steps: Number of interpolation steps
        
    Returns:
        dict: Interpolation results
    """
    model.eval()
    
    # Get molecules
    mol1 = dataset[mol_idx1]
    mol2 = dataset[mol_idx2]
    
    coords1 = mol1['coordinates'].unsqueeze(0).to(device)
    coords2 = mol2['coordinates'].unsqueeze(0).to(device)
    mask1 = mol1['mask'].unsqueeze(0).to(device)
    mask2 = mol2['mask'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Interpolate
        interpolated = model.interpolate(
            coords1.squeeze(0), coords2.squeeze(0),
            mask1.squeeze(0), mask2.squeeze(0),
            num_steps
        )
        
        # Compute RMSD sequence
        rmsd_to_mol1 = []
        rmsd_to_mol2 = []
        
        for i, coords_interp in enumerate(interpolated):
            rmsd1 = compute_rmsd(coords_interp.unsqueeze(0), coords1, mask1)
            rmsd2 = compute_rmsd(coords_interp.unsqueeze(0), coords2, mask2)
            rmsd_to_mol1.append(rmsd1.item())
            rmsd_to_mol2.append(rmsd2.item())
    
    return {
        'interpolated_coords': interpolated.cpu(),
        'rmsd_to_mol1': rmsd_to_mol1,
        'rmsd_to_mol2': rmsd_to_mol2,
        'mol1_coords': coords1.cpu(),
        'mol2_coords': coords2.cpu(),
        'mol1_mask': mask1.cpu(),
        'mol2_mask': mask2.cpu()
    }

def create_plots(eval_results, latent_results, interp_results, output_dir):
    """
    Create evaluation plots.
    
    Args:
        eval_results: Reconstruction evaluation results
        latent_results: Latent space analysis results
        interp_results: Interpolation results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    
    # 1. RMSD distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eval_results['rmsd_all'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(eval_results['rmsd_mean'], color='red', linestyle='--', 
                label=f'Mean: {eval_results["rmsd_mean"]:.3f} Å')
    plt.axvline(eval_results['rmsd_median'], color='green', linestyle='--',
                label=f'Median: {eval_results["rmsd_median"]:.3f} Å')
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Frequency')
    plt.title('Reconstruction RMSD Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'rmsd_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Latent space PCA
    if latent_results['pca'] is not None:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_results['pca'][:, 0], latent_results['pca'][:, 1],
                            c=latent_results['molecular_features'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Number of Atoms')
        plt.xlabel(f'PC1 ({latent_results["pca_explained_variance"][0]:.1%} variance)')
        plt.ylabel(f'PC2 ({latent_results["pca_explained_variance"][1]:.1%} variance)')
        plt.title('Latent Space PCA Visualization')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'latent_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. t-SNE plot (if available)
    if latent_results['tsne'] is not None:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_results['tsne'][:, 0], latent_results['tsne'][:, 1],
                            c=latent_results['molecular_features'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Number of Atoms')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Latent Space t-SNE Visualization')
        plt.savefig(output_dir / 'latent_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Interpolation RMSD
    if interp_results:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        steps = range(len(interp_results['rmsd_to_mol1']))
        plt.plot(steps, interp_results['rmsd_to_mol1'], 'b-o', label='RMSD to Molecule 1')
        plt.plot(steps, interp_results['rmsd_to_mol2'], 'r-o', label='RMSD to Molecule 2')
        plt.xlabel('Interpolation Step')
        plt.ylabel('RMSD (Å)')
        plt.title('Interpolation RMSD Sequence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        smoothness = np.diff(interp_results['rmsd_to_mol1'])
        plt.plot(smoothness, 'g-o')
        plt.xlabel('Step')
        plt.ylabel('RMSD Change')
        plt.title('Interpolation Smoothness')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'interpolation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        ConforMorphVAE: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    model_state = checkpoint['model_state_dict']
    
    # Infer model architecture from state dict
    # This is a simplified approach - in practice, you'd save config with checkpoint
    max_atoms = 50  # Default, should be saved in config
    latent_dim = list(model_state.values())[0].shape[0] if 'fc_mu.weight' in str(list(model_state.keys())[0]) else 64
    
    model = ConforMorphVAE(max_atoms=max_atoms, latent_dim=latent_dim)
    model.load_state_dict(model_state)
    model.to(device)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate ConforMorph-VAE")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/processed/", help="Data directory")
    parser.add_argument("--output", type=str, default="evaluation/", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_interpolations", type=int, default=5, help="Number of interpolation pairs")
    parser.add_argument("--latent_samples", type=int, default=1000, help="Samples for latent analysis")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from: {args.model}")
    model = load_model(args.model, device)
    
    # Load test dataset
    test_dataset = MolecularDataset(Path(args.data) / "test.pt")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Evaluate reconstruction
    logger.info("Evaluating reconstruction quality...")
    eval_results = evaluate_reconstruction(model, test_loader, device)
    
    # Latent space analysis
    logger.info("Analyzing latent space...")
    latent_results = latent_space_analysis(model, test_loader, device, args.latent_samples)
    
    # Interpolation analysis
    logger.info("Performing interpolation analysis...")
    interp_results = None
    if len(test_dataset) >= 2:
        # Perform multiple interpolations
        interp_results = interpolate_molecules(model, test_dataset, device, 0, 1)
    
    # Create plots
    logger.info("Creating plots...")
    create_plots(eval_results, latent_results, interp_results, output_dir)
    
    # Save summary statistics
    summary = {
        'reconstruction_metrics': {
            'rmsd_mean': float(eval_results['rmsd_mean']),
            'rmsd_std': float(eval_results['rmsd_std']),
            'rmsd_median': float(eval_results['rmsd_median']),
            'rmsd_q75': float(eval_results['rmsd_q75']),
            'rmsd_q95': float(eval_results['rmsd_q95']),
            'recon_loss_mean': float(eval_results['recon_loss_mean']),
            'kld_loss_mean': float(eval_results['kld_loss_mean'])
        },
        'latent_space_metrics': {
            'pca_variance_explained': latent_results['pca_explained_variance'].tolist(),
            'latent_dimension': latent_results['latent_vectors'].shape[1]
        }
    }
    
    # Save summary
    import json
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Reconstruction RMSD: {eval_results['rmsd_mean']:.3f} ± {eval_results['rmsd_std']:.3f} Å")
    logger.info(f"Median RMSD: {eval_results['rmsd_median']:.3f} Å")
    logger.info(f"95th percentile RMSD: {eval_results['rmsd_q95']:.3f} Å")
    logger.info(f"Reconstruction Loss: {eval_results['recon_loss_mean']:.3f}")
    logger.info(f"KLD Loss: {eval_results['kld_loss_mean']:.3f}")
    logger.info(f"PCA explained variance: {latent_results['pca_explained_variance'][:2]}")
    logger.info("="*50)
    
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()