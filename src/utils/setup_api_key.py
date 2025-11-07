"""
Helper script to set up API keys for the project.

This script helps you configure your API keys by creating a .env file
or providing instructions on how to set environment variables.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / '.env'


def setup_xai_api_key():
    """Interactive setup for xAI API key."""
    print("=" * 60)
    print("xAI API Key Setup")
    print("=" * 60)
    print()
    
    # Check if .env file already exists
    if ENV_FILE.exists():
        print(f"Found existing .env file at: {ENV_FILE}")
        response = input("Do you want to update it? (y/n): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Check if key is already set in environment
    existing_key = os.getenv("XAI_API_KEY")
    if existing_key:
        print(f"Found XAI_API_KEY in environment variable.")
        response = input("Do you want to save it to .env file? (y/n): ").strip().lower()
        if response == 'y':
            api_key = existing_key
        else:
            api_key = input("Enter your xAI API key: ").strip()
    else:
        api_key = input("Enter your xAI API key (get it from https://x.ai/api): ").strip()
    
    if not api_key:
        print("No API key provided. Setup cancelled.")
        return
    
    # Write to .env file
    try:
        # Read existing .env file if it exists
        env_lines = []
        if ENV_FILE.exists():
            with open(ENV_FILE, 'r') as f:
                env_lines = f.readlines()
        
        # Update or add XAI_API_KEY
        key_found = False
        updated_lines = []
        for line in env_lines:
            if line.startswith('XAI_API_KEY='):
                updated_lines.append(f'XAI_API_KEY={api_key}\n')
                key_found = True
            else:
                updated_lines.append(line)
        
        if not key_found:
            updated_lines.append(f'XAI_API_KEY={api_key}\n')
        
        # Write back to file
        with open(ENV_FILE, 'w') as f:
            f.writelines(updated_lines)
        
        print()
        print(f"✓ API key saved to {ENV_FILE}")
        print()
        print("Next steps:")
        print("1. Make sure python-dotenv is installed: pip install python-dotenv")
        print("2. Run your script again")
        print()
        print("Note: The .env file is already in .gitignore, so your key won't be committed.")
        
    except Exception as e:
        print(f"Error saving API key: {e}")
        print()
        print("You can manually create a .env file in the project root with:")
        print(f"XAI_API_KEY={api_key}")


if __name__ == "__main__":
    setup_xai_api_key()

