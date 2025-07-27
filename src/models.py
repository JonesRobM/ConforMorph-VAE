"""
VAE model architectures for ConforMorph-VAE.
Defines encoder, decoder, and complete VAE models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MolecularEncoder(nn.Module):
    """
    MLP encoder that maps molecular coordinates to latent space.
    """
    
    def __init__(self, max_atoms=50, latent_dim=64, hidden_dims=[512, 256, 128]):
        """
        Args:
            max_atoms (int): Maximum number of atoms (input size)
            latent_dim (int): Latent space dimensionality
            hidden_dims (list): Hidden layer dimensions
        """
        super().__init__()
        
        self.max_atoms = max_atoms
        self.latent_dim = latent_dim
        
        # Input dimension: max_atoms * 3 (x, y, z coordinates)
        input_dim = max_atoms * 3
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, mask=None):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input coordinates [batch_size, max_atoms, 3]
            mask (torch.Tensor): Atom mask [batch_size, max_atoms]
            
        Returns:
            tuple: (mu, logvar) - latent mean and log-variance
        """
        batch_size = x.size(0)
        
        # Apply mask to zero out padding atoms
        if mask is not None:
            x = x * mask.unsqueeze(-1)  # [batch_size, max_atoms, 3]
        
        # Flatten coordinates
        x_flat = x.view(batch_size, -1)  # [batch_size, max_atoms * 3]
        
        # Encode
        h = self.encoder(x_flat)
        
        # Project to latent space
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

class MolecularDecoder(nn.Module):
    """
    MLP decoder that maps latent vectors back to molecular coordinates.
    """
    
    def __init__(self, latent_dim=64, max_atoms=50, hidden_dims=[128, 256, 512]):
        """
        Args:
            latent_dim (int): Latent space dimensionality
            max_atoms (int): Maximum number of atoms (output size)
            hidden_dims (list): Hidden layer dimensions
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        
        # Output dimension: max_atoms * 3 (x, y, z coordinates)
        output_dim = max_atoms * 3
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final output layer (no activation - coordinates can be negative)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        """
        Forward pass through decoder.
        
        Args:
            z (torch.Tensor): Latent vectors [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Reconstructed coordinates [batch_size, max_atoms, 3]
        """
        batch_size = z.size(0)
        
        # Decode
        x_flat = self.decoder(z)  # [batch_size, max_atoms * 3]
        
        # Reshape to coordinate format
        x_recon = x_flat.view(batch_size, self.max_atoms, 3)
        
        return x_recon

class ConforMorphVAE(nn.Module):
    """
    Complete Variational Autoencoder for molecular conformations.
    """
    
    def __init__(self, max_atoms=50, latent_dim=64, 
                 encoder_hidden=[512, 256, 128], 
                 decoder_hidden=[128, 256, 512]):
        """
        Args:
            max_atoms (int): Maximum number of atoms
            latent_dim (int): Latent space dimensionality
            encoder_hidden (list): Encoder hidden dimensions
            decoder_hidden (list): Decoder hidden dimensions
        """
        super().__init__()
        
        self.max_atoms = max_atoms
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder
        self.encoder = MolecularEncoder(
            max_atoms=max_atoms,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden
        )
        
        self.decoder = MolecularDecoder(
            latent_dim=latent_dim,
            max_atoms=max_atoms,
            hidden_dims=decoder_hidden
        )
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        
        Args:
            mu (torch.Tensor): Latent means
            logvar (torch.Tensor): Latent log-variances
            
        Returns:
            torch.Tensor: Sampled latent vectors
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, mask=None):
        """
        Forward pass through complete VAE.
        
        Args:
            x (torch.Tensor): Input coordinates [batch_size, max_atoms, 3]
            mask (torch.Tensor): Atom mask [batch_size, max_atoms]
            
        Returns:
            dict: Dictionary containing:
                - recon_x: Reconstructed coordinates
                - mu: Latent means
                - logvar: Latent log-variances
                - z: Sampled latent vectors
        """
        # Encode
        mu, logvar = self.encoder(x, mask)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decoder(z)
        
        return {
            'recon_x': recon_x,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def sample(self, num_samples, device='cpu'):
        """
        Generate new molecular conformations by sampling from latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to use for generation
            
        Returns:
            torch.Tensor: Generated coordinates [num_samples, max_atoms, 3]
        """
        self.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Decode
            generated = self.decoder(z)
            
        return generated
    
    def interpolate(self, x1, x2, mask1=None, mask2=None, num_steps=10):
        """
        Interpolate between two molecular conformations in latent space.
        
        Args:
            x1, x2 (torch.Tensor): Input coordinates to interpolate between
            mask1, mask2 (torch.Tensor): Corresponding atom masks
            num_steps (int): Number of interpolation steps
            
        Returns:
            torch.Tensor: Interpolated coordinates [num_steps, max_atoms, 3]
        """
        self.eval()
        with torch.no_grad():
            # Encode both molecules
            mu1, _ = self.encoder(x1.unsqueeze(0), mask1.unsqueeze(0) if mask1 is not None else None)
            mu2, _ = self.encoder(x2.unsqueeze(0), mask2.unsqueeze(0) if mask2 is not None else None)
            
            # Create interpolation weights
            alphas = torch.linspace(0, 1, num_steps, device=x1.device)
            
            # Interpolate in latent space
            interpolated_coords = []
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decoder(z_interp)
                interpolated_coords.append(x_interp.squeeze(0))
            
            return torch.stack(interpolated_coords)

def vae_loss(recon_x, x, mu, logvar, mask=None, beta=1.0):
    """
    VAE loss function combining reconstruction and KLD terms.
    
    Args:
        recon_x (torch.Tensor): Reconstructed coordinates
        x (torch.Tensor): Original coordinates
        mu (torch.Tensor): Latent means
        logvar (torch.Tensor): Latent log-variances
        mask (torch.Tensor): Atom mask
        beta (float): KLD weighting factor
        
    Returns:
        dict: Loss components and total loss
    """
    # Reconstruction loss (MSE)
    if mask is not None:
        # Only compute loss for real atoms (not padding)
        mask_expanded = mask.unsqueeze(-1)  # [batch_size, max_atoms, 1]
        recon_loss = F.mse_loss(recon_x * mask_expanded, x * mask_expanded, reduction='sum')
        recon_loss = recon_loss / mask.sum()  # Normalize by number of real atoms
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KLD loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # Total loss
    total_loss = recon_loss + beta * kld_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kld_loss': kld_loss
    }

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)