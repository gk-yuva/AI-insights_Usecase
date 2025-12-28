"""
Configuration Module
Loads API credentials and settings from environment variables
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Configuration class for API credentials"""
    
    # Upstox API Configuration
    UPSTOX_API_KEY = os.getenv('UPSTOX_API_KEY')
    UPSTOX_API_SECRET = os.getenv('UPSTOX_API_SECRET')
    UPSTOX_REDIRECT_URI = os.getenv('UPSTOX_REDIRECT_URI')
    UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
    UPSTOX_AUTH_CODE = os.getenv('UPSTOX_AUTH_CODE')
    
    @classmethod
    def validate_upstox_credentials(cls) -> bool:
        """
        Validate that required Upstox credentials are present
        
        Returns:
            True if credentials are valid, False otherwise
        """
        if not cls.UPSTOX_ACCESS_TOKEN:
            if not all([cls.UPSTOX_API_KEY, cls.UPSTOX_API_SECRET, cls.UPSTOX_REDIRECT_URI]):
                return False
        return True
    
    @classmethod
    def get_missing_credentials(cls) -> list:
        """
        Get list of missing credential keys
        
        Returns:
            List of missing credential names
        """
        missing = []
        
        if not cls.UPSTOX_ACCESS_TOKEN:
            if not cls.UPSTOX_API_KEY:
                missing.append('UPSTOX_API_KEY')
            if not cls.UPSTOX_API_SECRET:
                missing.append('UPSTOX_API_SECRET')
            if not cls.UPSTOX_REDIRECT_URI:
                missing.append('UPSTOX_REDIRECT_URI')
        
        return missing


def check_config():
    """Check and print configuration status"""
    print("Configuration Status:")
    print("=" * 60)
    
    if Config.validate_upstox_credentials():
        print("✓ Upstox API credentials configured")
    else:
        print("✗ Upstox API credentials missing")
        missing = Config.get_missing_credentials()
        print(f"  Missing: {', '.join(missing)}")
        print(f"\n  Please update .env file with your credentials")
        print(f"  See .env.template for reference")
    
    print("=" * 60)


if __name__ == "__main__":
    check_config()
