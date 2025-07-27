"""
PubChem molecular data scraper for ConforMorph-VAE.
Scrapes CIDs from index pages and downloads 3D SDF files.
"""

import argparse
import time
import requests
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_sample_cids():
    """
    Get a curated list of CIDs known to have 3D structures.
    These are common small molecules from drug databases.
    
    Returns:
        list: List of CID integers with confirmed 3D structures
    """
    # Common drug molecules and natural products with 3D structures
    sample_cids = [
        # Drugs
        2244,      # Aspirin
        3672,      # Ibuprofen
        4409,      # Caffeine
        5090,      # Morphine
        60823,     # Imatinib
        2519,      # Cocaine
        5280445,   # Resveratrol
        5280343,   # Quercetin
        445639,    # Sildenafil
        2662,      # Glucose
        
        # Amino acids
        5950,      # Alanine
        6322,      # Glycine
        6140,      # Phenylalanine
        6305,      # Tryptophan
        6274,      # Tyrosine
        
        # Nucleotides
        6083,      # Adenine
        1135,      # Guanine
        1174,      # Cytosine
        1135,      # Thymine
        
        # Common organic molecules
        241,       # Benzene
        996,       # Ethanol
        887,       # Methanol
        8078,      # Phenol
        7847,      # Aniline
        
        # More complex drugs
        135398744, # Remdesivir
        5311,      # Warfarin
        4080,      # Nicotine
        2244,      # Acetylsalicylic acid
        2078,      # Vanillin
    ]
    
    # Add ranges of consecutive CIDs that often have 3D structures
    # Small molecule range
    sample_cids.extend(range(1000, 1100))   # Small organics
    sample_cids.extend(range(5000, 5200))   # Drug-like molecules
    sample_cids.extend(range(10000, 10100)) # More complex structures
    
    return list(set(sample_cids))  # Remove duplicates

def fetch_cids_from_pubchem_api(query="aspirin", max_cids=100):
    """
    Use PubChem's REST API to search for compounds and get CIDs.
    
    Args:
        query (str): Search query
        max_cids (int): Maximum CIDs to return
        
    Returns:
        list: List of CID integers
    """
    try:
        # PubChem compound search API
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/cids/JSON"
        response = requests.get(search_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            return cids[:max_cids]
        else:
            logger.warning(f"API search failed for query: {query}")
            return []
            
    except Exception as e:
        logger.warning(f"API search error: {e}")
        return []

def fetch_cids(source_type="sample", query="drug", max_cids=500):
    """
    Get CIDs from various sources.
    
    Args:
        source_type (str): Type of source ('sample', 'api', 'file')
        query (str): Query for API search
        max_cids (int): Maximum CIDs to return
        
    Returns:
        list: List of CID integers
    """
    if source_type == "sample":
        cids = get_sample_cids()
        logger.info(f"Using curated sample CIDs: {len(cids)} total")
        return cids[:max_cids]
    
    elif source_type == "api":
        # Try multiple search terms for diversity
        search_terms = [query, "drug", "natural product", "vitamin", "hormone"]
        all_cids = []
        
        for term in search_terms:
            cids = fetch_cids_from_pubchem_api(term, max_cids // len(search_terms))
            all_cids.extend(cids)
            time.sleep(0.5)  # Rate limiting
            
            if len(all_cids) >= max_cids:
                break
        
        logger.info(f"Found {len(all_cids)} CIDs from API search")
        return list(set(all_cids))[:max_cids]
    
    else:
        logger.error(f"Unknown source type: {source_type}")
        return []

def download_sdf(cid, output_dir, max_retries=3):
    """
    Download 3D SDF file for a given CID via PUG-REST API.
    
    Args:
        cid (int): PubChem Compound ID
        output_dir (Path): Directory to save SDF files
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        bool: True if successful, False otherwise
    """
    output_file = output_dir / f"{cid}.sdf"
    
    # Skip if file already exists
    if output_file.exists():
        return True
    
    # PUG-REST URL for 3D SDF
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=3d"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                with open(output_file, 'w') as f:
                    f.write(response.text)
                return True
            elif response.status_code == 404:
                logger.warning(f"No 3D structure available for CID {cid}")
                return False
            else:
                response.raise_for_status()
                
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for CID {cid}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)  # Progressive delay
            else:
                logger.error(f"Failed to download CID {cid} after {max_retries} attempts")
                return False
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Download PubChem SDF files")
    parser.add_argument("--source", type=str, default="sample", 
                       choices=["sample", "api"], 
                       help="Source of CIDs: 'sample' for curated list, 'api' for PubChem search")
    parser.add_argument("--query", type=str, default="drug", 
                       help="Search query for API source")
    parser.add_argument("--output", type=str, default="data/raw/", help="Output directory")
    parser.add_argument("--max_molecules", type=int, default=500, help="Maximum molecules to download")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between downloads (seconds)")
    parser.add_argument("--test_mode", action="store_true", help="Download only 10 molecules for testing")
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.max_molecules = 10
        logger.info("Test mode: downloading only 10 molecules")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting data acquisition")
    logger.info(f"Source: {args.source}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max molecules: {args.max_molecules}")
    
    # Get CIDs
    all_cids = fetch_cids(args.source, args.query, args.max_molecules)
    
    if not all_cids:
        logger.error("No CIDs found! Check your source configuration.")
        return
    
    logger.info(f"Retrieved {len(all_cids)} CIDs")
    
    # Download SDF files
    successful_downloads = 0
    failed_downloads = 0
    
    for cid in tqdm(all_cids, desc="Downloading SDF files"):
        if download_sdf(cid, output_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        time.sleep(args.delay)
        
        # Stop early if test mode and we have enough
        if args.test_mode and successful_downloads >= 10:
            break
    
    logger.info(f"Download complete: {successful_downloads} successful, {failed_downloads} failed")
    logger.info(f"Files saved to: {output_dir}")
    
    if successful_downloads == 0:
        logger.error("No files downloaded successfully! Check network connection and CID validity.")
    elif successful_downloads < 10:
        logger.warning("Very few files downloaded. Consider using --source sample for reliable test data.")

if __name__ == "__main__":
    main()