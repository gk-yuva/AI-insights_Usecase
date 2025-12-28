"""
Test Upstox Historical Data API - Version 2
Testing without from_date parameter
"""

import upstox_client
from config import Config
from datetime import datetime, timedelta

def test_historical_simple():
    """Test fetching historical candle data with simpler parameters"""
    print("=" * 70)
    print("UPSTOX HISTORICAL DATA TEST - SIMPLE")
    print("=" * 70)
    
    try:
        # Initialize API client
        configuration = upstox_client.Configuration()
        configuration.access_token = Config.UPSTOX_ACCESS_TOKEN
        
        api_client = upstox_client.ApiClient(configuration)
        history_api = upstox_client.HistoryApi(api_client)
        
        instrument_key = "NSE_EQ|NATIONALUM"
        
        print(f"\nFetching historical data for: {instrument_key}")
        
        # Try getting data without from_date (just to_date)
        try:
            print(f"\nMethod 1: Using only to_date...")
            end_date = datetime.now()
            to_date_str = end_date.strftime('%Y-%m-%d')
            
            response = history_api.get_historical_candle_data(
                instrument_key=instrument_key,
                interval='day',
                to_date=to_date_str,
                api_version='2.0'
            )
            
            print(f"  ✓ SUCCESS!")
            
            if hasattr(response, 'data') and hasattr(response.data, 'candles'):
                candles = response.data.candles
                print(f"  Retrieved {len(candles)} candles")
                if candles and len(candles) > 0:
                    print(f"  First candle: {candles[0]}")
                    print(f"  Last candle: {candles[-1]}")
                    print(f"\n  Candle format: [timestamp, open, high, low, close, volume, oi]")
            else:
                print(f"  Response: {response}")
                
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:300]}")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    test_historical_simple()
