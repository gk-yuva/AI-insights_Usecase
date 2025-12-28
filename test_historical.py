"""
Test Upstox Historical Data API
"""

import upstox_client
from config import Config
from datetime import datetime, timedelta

def test_historical_data():
    """Test fetching historical candle data"""
    print("=" * 70)
    print("UPSTOX HISTORICAL DATA TEST")
    print("=" * 70)
    
    try:
        # Initialize API client
        configuration = upstox_client.Configuration()
        configuration.access_token = Config.UPSTOX_ACCESS_TOKEN
        
        api_client = upstox_client.ApiClient(configuration)
        history_api = upstox_client.HistoryApi(api_client)
        
        # Set up dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        instrument_key = "NSE_EQ|NATIONALUM"
        
        print(f"\nFetching historical data for: {instrument_key}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        
        # Try different date formats
        date_formats = [
            (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            (start_date.strftime('%d-%m-%Y'), end_date.strftime('%d-%m-%Y')),
        ]
        
        for from_date, to_date in date_formats:
            try:
                print(f"\nTrying date format: {from_date} to {to_date}")
                
                response = history_api.get_historical_candle_data1(
                    instrument_key=instrument_key,
                    interval='day',
                    to_date=to_date,
                    from_date=from_date,
                    api_version='2.0'
                )
                
                print(f"  ✓ SUCCESS!")
                
                if hasattr(response, 'data') and hasattr(response.data, 'candles'):
                    candles = response.data.candles
                    print(f"  Retrieved {len(candles)} candles")
                    if candles:
                        print(f"  First candle: {candles[0]}")
                        print(f"  Last candle: {candles[-1]}")
                    break
                else:
                    print(f"  Response: {response}")
                    
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:200]}")
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    test_historical_data()
