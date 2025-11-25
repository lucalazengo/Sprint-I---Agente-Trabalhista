"""
Configuration utility for managing API keys, environment variables, and project paths.

This module serves as the Single Source of Truth (SSOT) for the application configuration.
It dynamically resolves paths relative to the project root, ensuring portability.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Config")

# --- Project Paths ---
# Resolves to: .../Sprint-I---Agente-Trabalhista/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
INDICES_DIR = PROCESSED_DATA_DIR / "indices"

# Artifact Paths
PATH_BM25_INDEX = INDICES_DIR / "clt_bm25_bge-m3.pkl"
PATH_CHROMA_DB = INDICES_DIR / "chroma_db_bge_m3"
PATH_MAPPING_FILE = PROCESSED_DATA_DIR / "clt_chunks.json"

# Secrets
SECRETS_DIR = PROJECT_ROOT / "src" / "backend" / "secrets"
ENV_FILE = SECRETS_DIR / ".env"

# --- Model Configuration ---
MODEL_EMBEDDING = 'BAAI/bge-m3'
CHROMA_COLLECTION_NAME = "clt_bge_m3"
LLM_MODEL_NAME = 'sabia-3'

def load_environment() -> bool:
    """
    Load environment variables from .env file.
    Prioritizes the specific secrets/.env file, then falls back to project root .env.
    """
    env_loaded = False
    
    # Try specific backend secrets location first
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
        env_loaded = True
        logger.info(f"Loaded configuration from {ENV_FILE}")
    
    # Try root .env as fallback
    root_env = PROJECT_ROOT / ".env"
    if not env_loaded and root_env.exists():
        load_dotenv(root_env)
        env_loaded = True
        logger.info(f"Loaded configuration from {root_env}")
        
    if not env_loaded and not os.getenv("OPENAI_API_KEY"):
        logger.warning("No .env file found and OPENAI_API_KEY not set in environment.")
        
    return env_loaded

def get_api_key(key_name: str = "OPENAI_API_KEY", required: bool = True) -> Optional[str]:
    """
    Get an API key from environment variables.
    """
    load_environment()
    api_key = os.getenv(key_name)
    
    if not api_key and required:
        raise ValueError(
            f"Missing required environment variable: {key_name}\n"
            f"Please check your .env file at {ENV_FILE}"
        )
    return api_key

# Initialize environment on import
load_environment()
