"""
Upstox Token Generator
Generates access token for Upstox API using OAuth flow
"""

import webbrowser
from urllib.parse import urlparse, parse_qs
from config import Config
import requests


def generate_authorization_url():
    """Generate Upstox authorization URL"""
    base_url = "https://api.upstox.com/v2/login/authorization/dialog"
    params = {
        'response_type': 'code',
        'client_id': Config.UPSTOX_API_KEY,
        'redirect_uri': Config.UPSTOX_REDIRECT_URI
    }
    
    auth_url = f"{base_url}?response_type={params['response_type']}&client_id={params['client_id']}&redirect_uri={params['redirect_uri']}"
    return auth_url


def exchange_code_for_token(auth_code: str) -> str:
    """
    Exchange authorization code for access token
    
    Args:
        auth_code: Authorization code from redirect
        
    Returns:
        Access token
    """
    token_url = "https://api.upstox.com/v2/login/authorization/token"
    
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    
    data = {
        'code': auth_code,
        'client_id': Config.UPSTOX_API_KEY,
        'client_secret': Config.UPSTOX_API_SECRET,
        'redirect_uri': Config.UPSTOX_REDIRECT_URI,
        'grant_type': 'authorization_code'
    }
    
    response = requests.post(token_url, headers=headers, data=data)
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data.get('access_token')
    else:
        raise Exception(f"Token exchange failed: {response.text}")


def save_token_to_env(access_token: str):
    """Save access token to .env file"""
    import os
    from pathlib import Path
    
    env_path = Path(__file__).parent / '.env'
    
    # Read existing .env content
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
    else:
        # Create from template
        template_path = Path(__file__).parent / '.env.template'
        with open(template_path, 'r') as f:
            lines = f.readlines()
    
    # Update access token line
    updated_lines = []
    token_found = False
    
    for line in lines:
        if line.startswith('UPSTOX_ACCESS_TOKEN='):
            updated_lines.append(f'UPSTOX_ACCESS_TOKEN={access_token}\n')
            token_found = True
        else:
            updated_lines.append(line)
    
    # Add if not found
    if not token_found:
        updated_lines.append(f'\nUPSTOX_ACCESS_TOKEN={access_token}\n')
    
    # Write back to .env
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"✓ Access token saved to {env_path}")


def main():
    """Main token generation flow"""
    print("=" * 70)
    print("UPSTOX ACCESS TOKEN GENERATOR")
    print("=" * 70)
    
    # Check if credentials are configured
    if not all([Config.UPSTOX_API_KEY, Config.UPSTOX_API_SECRET, Config.UPSTOX_REDIRECT_URI]):
        print("\n❌ Error: Missing Upstox credentials in .env file")
        print("\nPlease configure the following in .env:")
        if not Config.UPSTOX_API_KEY:
            print("  - UPSTOX_API_KEY")
        if not Config.UPSTOX_API_SECRET:
            print("  - UPSTOX_API_SECRET")
        if not Config.UPSTOX_REDIRECT_URI:
            print("  - UPSTOX_REDIRECT_URI")
        print("\nSee .env.template for reference")
        return
    
    print("\nStep 1: Opening Upstox authorization page in your browser...")
    auth_url = generate_authorization_url()
    print(f"\nIf browser doesn't open, visit this URL:\n{auth_url}\n")
    
    webbrowser.open(auth_url)
    
    print("\nStep 2: After authorizing, you'll be redirected to your redirect URI")
    print("        Copy the entire redirect URL from your browser's address bar")
    
    redirect_url = input("\nPaste the redirect URL here: ").strip()
    
    # Parse the code from redirect URL
    try:
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        auth_code = params.get('code', [None])[0]
        
        if not auth_code:
            print("\n❌ Error: No authorization code found in URL")
            print("Make sure you copied the complete URL after authorization")
            return
        
        print(f"\n✓ Authorization code extracted: {auth_code[:20]}...")
        
        print("\nStep 3: Exchanging code for access token...")
        access_token = exchange_code_for_token(auth_code)
        
        print(f"✓ Access token obtained: {access_token[:20]}...")
        
        print("\nStep 4: Saving token to .env file...")
        save_token_to_env(access_token)
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Upstox API is now configured")
        print("=" * 70)
        print("\nYou can now run: python portfolio_analyzer.py")
        print("\nNote: Access tokens typically expire after 24 hours.")
        print("      Re-run this script when you need a fresh token.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease try again or check your credentials")


if __name__ == "__main__":
    main()
