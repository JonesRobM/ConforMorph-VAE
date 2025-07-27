"""
PubChem molecular data scraper for ConforMorph-VAE.
Scrapes CIDs from index pages and downloads 3D SDF files.
"""

import argparse
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_cids(page_url, max_retries=3):
    """
    Scrape CID list from a PubChem index page.
    
    Args:
        page_url (str): URL of the PubChem page to scrape
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        list: List of CID integers
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(page_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find CID links - adjust selector based on PubChem structure
            cid_links = soup.find_all('a', href=lambda x: x and '/compound/' in x)
            cids = []
            
            for link in cid_links:
                href = link.get('href')
                if '/compound/' in href:
                    try:
                        cid = int(href.split('/compound/')[-1].split('#')[0])
                        cids.append(cid)
                    except (ValueError, IndexError):
                        continue
            
            logger.info(f"Found {len(cids)} CIDs on page: {page_url}")
            return list(set(cids))  # Remove duplicates
            
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {page_url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to fetch CIDs from {page_url} after {max_retries} attempts")
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
    parser = argparse.ArgumentParser(description="Scrape PubChem CIDs and download SDF files")
    parser.add_argument("--start_page", type=int, default=1, help="Starting page index")
    parser.add_argument("--end_page", type=int, default=5, help="Ending page index")
    parser.add_argument("--output", type=str, default="data/raw/", help="Output directory")
    parser.add_argument("--max_molecules", type=int, default=1000, help="Maximum molecules to download")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between downloads (seconds)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting scraper: pages {args.start_page}-{args.end_page}")
    logger.info(f"Output directory: {output_dir}")
    
    all_cids = []
    
    # Collect CIDs from multiple pages
    for page_num in range(args.start_page, args.end_page + 1):
        # Example PubChem URL pattern - adjust based on actual structure
        page_url = f"https://pubchem.ncbi.nlm.nih.gov/compound?ctype=3D&page={page_num}"
        cids = fetch_cids(page_url)
        all_cids.extend(cids)
        
        if len(all_cids) >= args.max_molecules:
            break
        
        time.sleep(args.delay)
    
    # Limit to max_molecules
    all_cids = list(set(all_cids))[:args.max_molecules]
    logger.info(f"Collected {len(all_cids)} unique CIDs")
    
    # Download SDF files
    successful_downloads = 0
    failed_downloads = 0
    
    for cid in tqdm(all_cids, desc="Downloading SDF files"):
        if download_sdf(cid, output_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        time.sleep(args.delay)
    
    logger.info(f"Download complete: {successful_downloads} successful, {failed_downloads} failed")
    logger.info(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()