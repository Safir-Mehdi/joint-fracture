"""
Project path resolution and environment configuration utilities.

Purpose:
    Dynamically locates the `joint-fracture` project root directory across both
    standard Python scripts and Jupyter Notebooks, loads `.env` variables, and
    resolves validated absolute paths for project files and folders.

Usage:
    from src.utils.paths import get_path
    data_dir = get_path("data/raw")
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def _find_project_root() -> Path:
    """Finds the 'joint-fracture' root directory dynamically in both scripts and notebooks."""
    # 1. Check if __file__ exists (Standard Python Script)
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2] # Go up to joint-fracture/
        
    # 2. Fallback for Jupyter Notebooks
    current_path = Path('.').resolve()
    if current_path.name == "joint-fracture":
        return current_path
    
    # Climb up the tree until we find the root folder name
    for parent in current_path.parents:
        if parent.name == "joint-fracture":
            return parent
            
    raise FileNotFoundError("Could not find the 'joint-fracture' project root directory.")

# Automatically find and load the correct .env file
PROJECT_DIR = _find_project_root()
dotenv_path = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path) # Removed the second empty load_dotenv()

def get_path(sub_path: str | Path) -> Path:
    """
    Returns an absolute path to any file or folder by combining 
    the PROJECT_ROOT from the .env file with the target subpath.
    """
    root_env = os.getenv("PROJECT_ROOT")
    
    if not root_env:
        raise ValueError(f"PROJECT_ROOT is not set in the .env file. Looked at: {dotenv_path}")
        
    target_path = (Path(root_env) / sub_path).resolve()
    
    if not target_path.exists():
        raise FileNotFoundError(f"Path does not exist: {target_path}")
        
    return target_path
