"""
Quick verification script to test if the API key is loaded correctly.
"""

import sys
from pathlib import Path

# Add the utils directory to the path
utils_dir = Path(__file__).parent
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

try:
    from config import get_xai_api_key, load_environment, ENV_FILE
    
    print("=" * 60)
    print("API Key Verification")
    print("=" * 60)
    print()
    
    # Check if .env file exists
    if ENV_FILE.exists():
        print(f"✓ Found .env file at: {ENV_FILE}")
    else:
        print(f"✗ .env file not found at: {ENV_FILE}")
        print("  Run: python src/utils/setup_api_key.py")
        sys.exit(1)
    
    # Try to load environment
    env_loaded = load_environment()
    if env_loaded:
        print("✓ Successfully loaded .env file")
    else:
        print("⚠ Could not load .env file (python-dotenv may not be installed)")
        print("  Install with: pip install python-dotenv")
    
    # Try to get API key
    try:
        api_key = get_xai_api_key()
        # Mask the key for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ API key loaded successfully: {masked_key}")
        print()
        print("Setup is complete! You can now run api_test.py")
        print()
    except ValueError as e:
        print(f"✗ Error loading API key:")
        print(f"  {e}")
        sys.exit(1)
        
except ImportError as e:
    print(f"✗ Error importing config module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

