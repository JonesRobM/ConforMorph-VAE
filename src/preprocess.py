"""
Molecular preprocessing for ConforMorph-VAE.
Converts SDF files to fixed-size coordinate tensors.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_molecule(mol):
    """
    Validate molecule for 3D structure and basic properties.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        bool: True if valid, False otherwise
    """
    if mol is None:
        return False
    
    # Check if molecule has 3D coordinates
    conf = mol.GetConformer(0) if mol.GetNumConformers() > 0 else None
    if conf is None:
        return False
    
    # Check for reasonable number of atoms
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 3 or num_atoms > 100:  # Reasonable bounds
        return False
    
    # Basic sanitisation
    try:
        Chem.SanitizeMol(mol)
    except:
        return False
    
    return True

def extract_coordinates(mol, max_atoms=50):
    """
    Extract 3D coordinates from molecule and pad/truncate to fixed size.
    
    Args:
        mol: RDKit molecule object
        max_atoms (int): Maximum number of atoms (padding size)
        
    Returns:
        torch.Tensor: Coordinates tensor of shape (max_atoms, 3)
        torch.Tensor: Atom mask tensor of shape (max_atoms,) - 1 for real atoms, 0 for padding
        list: Atomic numbers
    """
    conf = mol.GetConformer(0)
    num_atoms = mol.GetNumAtoms()
    
    # Initialize tensors
    coords = torch.zeros(max_atoms, 3, dtype=torch.float32)
    mask = torch.zeros(max_atoms, dtype=torch.bool)
    atomic_numbers = [0] * max_atoms
    
    # Fill coordinates for actual atoms
    for i in range(min(num_atoms, max_atoms)):
        pos = conf.GetAtomPosition(i)
        coords[i] = torch.tensor([pos.x, pos.y, pos.z], dtype=torch.float32)
        mask[i] = True
        atomic_numbers[i] = mol.GetAtomWithIdx(i).GetAtomicNum()
    
    return coords, mask, atomic_numbers

def compute_molecular_features(mol):
    """
    Compute basic molecular descriptors.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        dict: Dictionary of molecular features
    """
    features = {
        'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': rdMolDescriptors.CalcNumRings(mol),
        'formal_charge': Chem.rdmolops.GetFormalCharge(mol)
    }
    return features

def process_sdf_file(sdf_path, max_atoms=50):
    """
    Process a single SDF file and extract coordinate tensor.
    
    Args:
        sdf_path (Path): Path to SDF file
        max_atoms (int): Maximum number of atoms
        
    Returns:
        dict or None: Processed molecule data or None if invalid
    """
    try:
        # Read molecule from SDF
        mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True, removeHs=False)
        
        if not validate_molecule(mol):
            return None
        
        # Extract coordinates and features
        coords, mask, atomic_numbers = extract_coordinates(mol, max_atoms)
        features = compute_molecular_features(mol)
        
        return {
            'cid': int(sdf_path.stem),
            'coordinates': coords,
            'mask': mask,
            'atomic_numbers': atomic_numbers,
            'features': features,
            'smiles': Chem.MolToSmiles(mol)
        }
        
    except Exception as e:
        logger.warning(f"Failed to process {sdf_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Preprocess SDF files to coordinate tensors")
    parser.add_argument("--input", type=str, default="data/raw/", help="Input directory with SDF files")
    parser.add_argument("--output", type=str, default="data/processed/", help="Output directory")
    parser.add_argument("--max_atoms", type=int, default=50, help="Maximum number of atoms (padding size)")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training set fraction")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation set fraction")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing SDF files from: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max atoms: {args.max_atoms}")
    
    # Find all SDF files
    sdf_files = list(input_dir.glob("*.sdf"))
    logger.info(f"Found {len(sdf_files)} SDF files")
    
    if len(sdf_files) == 0:
        logger.error("No SDF files found!")
        return
    
    # Process all molecules
    processed_molecules = []
    failed_count = 0
    
    for sdf_path in tqdm(sdf_files, desc="Processing molecules"):
        mol_data = process_sdf_file(sdf_path, args.max_atoms)
        if mol_data is not None:
            processed_molecules.append(mol_data)
        else:
            failed_count += 1
    
    logger.info(f"Successfully processed: {len(processed_molecules)} molecules")
    logger.info(f"Failed to process: {failed_count} molecules")
    
    if len(processed_molecules) == 0:
        logger.error("No valid molecules processed!")
        return
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(processed_molecules))
    
    n_train = int(len(processed_molecules) * args.train_split)
    n_val = int(len(processed_molecules) * args.val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create data splits
    splits = {
        'train': [processed_molecules[i] for i in train_indices],
        'val': [processed_molecules[i] for i in val_indices],
        'test': [processed_molecules[i] for i in test_indices]
    }
    
    # Save processed data
    for split_name, molecules in splits.items():
        if len(molecules) == 0:
            continue
            
        # Stack tensors
        coordinates = torch.stack([mol['coordinates'] for mol in molecules])
        masks = torch.stack([mol['mask'] for mol in molecules])
        
        # Save tensors
        torch.save({
            'coordinates': coordinates,
            'masks': masks,
            'cids': [mol['cid'] for mol in molecules],
            'atomic_numbers': [mol['atomic_numbers'] for mol in molecules],
            'smiles': [mol['smiles'] for mol in molecules],
            'features': [mol['features'] for mol in molecules]
        }, output_dir / f"{split_name}.pt")
        
        logger.info(f"Saved {split_name} set: {len(molecules)} molecules")
    
    # Save metadata
    metadata = {
        'max_atoms': args.max_atoms,
        'total_molecules': len(processed_molecules),
        'train_size': len(splits['train']),
        'val_size': len(splits['val']),
        'test_size': len(splits['test']),
        'failed_count': failed_count
    }
    
    torch.save(metadata, output_dir / "metadata.pt")
    logger.info(f"Saved metadata: {metadata}")
    
    # Create summary DataFrame
    features_df = pd.DataFrame([mol['features'] for mol in processed_molecules])
    features_df['cid'] = [mol['cid'] for mol in processed_molecules]
    features_df['smiles'] = [mol['smiles'] for mol in processed_molecules]
    
    summary_path = output_dir / "summary.csv"
    features_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to: {summary_path}")
    
    # Print statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Number of atoms: {features_df['num_atoms'].describe()}")
    logger.info(f"Molecular weight: {features_df['molecular_weight'].describe()}")

if __name__ == "__main__":
    main()