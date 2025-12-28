"""
Test Upstox API Connection and Instrument Keys
"""

import upstox_client
from config import Config

def test_upstox_connection():
    """Test basic Upstox API connection"""
    print("=" * 70)
    print("UPSTOX API CONNECTION TEST")
    print("=" * 70)
    
    try:
        # Initialize API client
        configuration = upstox_client.Configuration()
        configuration.access_token = Config.UPSTOX_ACCESS_TOKEN
        
        api_client = upstox_client.ApiClient(configuration)
        
        # Test with Market Quote API (simpler than historical)
        market_api = upstox_client.MarketQuoteApi(api_client)
        
        print("\n✓ API client initialized successfully")
        print(f"✓ Access token: {Config.UPSTOX_ACCESS_TOKEN[:20]}...")
        
        # Try a few different instrument key formats for NATIONALUM
        test_instruments = [
            'NSE_EQ|NATIONALUM',
            'NSE_EQ|INE139A01034',  # NATIONALUM ISIN if available
        ]
        
        print("\nTesting instrument keys:")
        print("-" * 70)
        
        for instrument_key in test_instruments:
            try:
                print(f"\nTrying: {instrument_key}")
                response = market_api.get_full_market_quote(
                    symbol=instrument_key,
                    api_version='2.0'
                )
                print(f"  ✓ SUCCESS! This instrument key works")
                print(f"  Response: {response}")
                break
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:100]}")
        
        # Try getting user profile to verify auth
        print("\n" + "-" * 70)
        print("Testing user authentication:")
        user_api = upstox_client.UserApi(api_client)
        profile = user_api.get_profile(api_version='2.0')
        print(f"✓ User Profile Retrieved:")
        print(f"  User ID: {profile.data.user_id if hasattr(profile, 'data') else 'N/A'}")
        print(f"  User Name: {profile.data.user_name if hasattr(profile, 'data') else 'N/A'}")
        
        print("\n" + "=" * 70)
        print("✅ UPSTOX API IS WORKING!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible issues:")
        print("  1. Access token may be expired (tokens expire daily)")
        print("  2. Invalid API credentials")
        print("  3. Network connectivity issues")
        print("\nSolution: Run 'python generate_upstox_token.py' to get a fresh token")


if __name__ == "__main__":
    test_upstox_connection()
