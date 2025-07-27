# ConforMorph-VAE

Variational Autoencoder for learning molecular conformation representations in 3D space.

## Overview

ConforMorph-VAE trains on molecular 3D coordinates to learn a compressed latent representation that captures conformational diversity. The model reconstructs molecular geometries and enables interpolation between conformations.

## Project Structure

```
ConforMorph-VAE/
├── data/
│   ├── raw/                # Downloaded SDF files
│   └── processed/          # Preprocessed tensors
├── src/
│   ├── scraper.py         # PubChem data acquisition via REST API
│   ├── preprocess.py      # RDKit processing and tensor conversion
│   ├── models.py          # VAE architecture definitions
│   ├── train.py           # Training loop and checkpointing
│   └── evaluate.py        # RMSD evaluation and interpolation
├── notebooks/             # Analysis and visualisation
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python ≥ 3.8
- PyTorch ≥ 1.9
- RDKit ≥ 2022.03
- requests, beautifulsoup4, numpy, pandas, matplotlib, tqdm

## Usage

### 1. Data Acquisition
```bash
# Download curated sample molecules (recommended for testing)
python src/scraper.py --source sample --max_molecules 100 --output data/raw/

# For quick testing (downloads only 10 molecules)
python src/scraper.py --test_mode

# Use PubChem API search (more molecules, less reliable)
python src/scraper.py --source api --query "drug" --max_molecules 500
```

### 2. Preprocessing
```bash
# Convert SDF to coordinate tensors
python src/preprocess.py --input data/raw/ --output data/processed/ --max_atoms 50
```

### 3. Training
```bash
# Train VAE model
python src/train.py --data data/processed/ --epochs 100 --batch_size 32 --latent_dim 64
```

### 4. Evaluation
```bash
# Evaluate reconstruction and interpolation
python src/evaluate.py --model checkpoints/model_best.pt --data data/processed/
```

## Model Architecture

**Encoder**: MLP mapping 3D coordinates → latent mean and log-variance
**Decoder**: MLP mapping latent vector → reconstructed coordinates
**Loss**: MSE reconstruction + KLD regularisation

**Input**: Fixed-size tensor (N_max, 3) with zero-padding for smaller molecules
**Output**: Reconstructed coordinates (N_max, 3)

## Expected Results

- **Reconstruction RMSD**: < 2.0 Å on test set
- **Training time**: ~30 minutes for 1000 molecules, 100 epochs
- **Latent interpolation**: Smooth transitions between similar conformations

## Data Pipeline

1. **Data Acquisition**: Use curated CID lists or PubChem API search for known compounds
2. **Download**: Fetch 3D SDF files via PUG-REST API
3. **Validation**: RDKit sanitisation and 3D coordinate verification
4. **Tensorisation**: Convert to PyTorch tensors with consistent padding
5. **Training**: VAE with configurable architecture and hyperparameters

## File Descriptions

- `scraper.py`: Data acquisition using curated CID lists and PubChem REST API
- `preprocess.py`: RDKit-based molecular processing and tensor conversion
- `models.py`: PyTorch VAE implementation with encoder/decoder modules
- `train.py`: Training loop with checkpointing and loss logging
- `evaluate.py`: RMSD calculation and latent space interpolation analysis

## Troubleshooting

**Common Issues**:
- RDKit installation: Use conda-forge channel
- Memory limits: Reduce batch size or max_atoms parameter
- Network timeouts: Use --test_mode for initial testing, adjust delay between downloads
- Convergence: Tune learning rate and KLD weighting
- No 3D structures: Some CIDs lack 3D conformations, use --source sample for reliable data

## Citation

[Add publication details when available]