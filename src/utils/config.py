"""
Configuration utility for managing API keys and environment variables.

This module provides a centralized way to load API keys and configuration
from multiple sources:
1. .env file in project root
2. Environment variables
3. secrets/ directory (for development)
"""

import os
from pathlib import Path
from typing import Optional

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Get project root directory (3 levels up from this file: src/utils/config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / '.env'
SECRETS_DIR = PROJECT_ROOT / 'secrets'


def load_environment() -> bool:
    """
    Load environment variables from .env file if it exists.
    
    Returns:
        bool: True if .env file was loaded, False otherwise
    """
    if not DOTENV_AVAILABLE:
        return False
    
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
        return True
    return False


def get_api_key(key_name: str, required: bool = True) -> Optional[str]:
    """
    Get an API key from environment variables or .env file.
    
    Args:
        key_name: Name of the environment variable (e.g., 'XAI_API_KEY')
        required: If True, raises ValueError if key is not found
        
    Returns:
        str: The API key value, or None if not found and not required
        
    Raises:
        ValueError: If required=True and key is not found
    """
    # Try to load .env file first
    load_environment()
    
    # Get from environment variable
    api_key = os.getenv(key_name)
    
    if api_key:
        return api_key
    
    # If not found and required, raise error with helpful message
    if required:
        raise ValueError(
            f"{key_name} environment variable not found.\n"
            f"Please set it in one of the following ways:\n"
            f"1. Create a .env file in the project root ({ENV_FILE}) with: {key_name}=your_key_here\n"
            f"2. Set it in your environment:\n"
            f"   - Windows CMD: set {key_name}=your_key_here\n"
            f"   - Windows PowerShell: $env:{key_name}='your_key_here'\n"
            f"   - Linux/Mac: export {key_name}=your_key_here\n"
            f"3. Install python-dotenv for .env file support: pip install python-dotenv"
        )
    
    return None


def get_xai_api_key() -> str:
    """
    Get the xAI API key.
    
    Returns:
        str: The xAI API key
        
    Raises:
        ValueError: If the API key is not found
    """
    return get_api_key("XAI_API_KEY", required=True)


def get_groq_api_key() -> str:
    """
    Get the Groq API key.
    
    Returns:
        str: The Groq API key
        
    Raises:
        ValueError: If the API key is not found
    """
    return get_api_key("GROQ_API_KEY", required=True)


# Load environment on import
if DOTENV_AVAILABLE:
    load_environment()

