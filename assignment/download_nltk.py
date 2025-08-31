#!/usr/bin/env python3
"""
NLTK Resources Downloader
=========================
Author: Ram Sapkota
Description: Downloads required NLTK data packages for text processing
"""

import nltk

def download_nltk_resources():
    """Download essential NLTK resources"""
    resources = ["punkt", "wordnet", "omw-1.4", "stopwords"]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Downloaded {resource}")
        except Exception as e:
            print(f"✗ Failed to download {resource}: {e}")
    
    print("NLTK resources setup complete")

if __name__ == "__main__":
    download_nltk_resources()