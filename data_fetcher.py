"""
Data Fetcher Module
Fetches historical price data for portfolio holdings
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import upstox_client
from upstox_client.rest import ApiException
from config import Config


class DataFetcher:
    """Fetch historical price data for portfolio instruments"""
    
    # Mapping for Indian stocks to NSE symbols
    TICKER_MAPPINGS = {
        'NATIONALUM': 'NATIONALUM',  # NSE symbol (without .NS for Upstox)
        'GOLD1': 'GOLD',  # Will try to find gold ETF or use yfinance fallback
        'NIFTY50': 'NIFTY 50',  # Nifty 50 index
        'NIFTYMETAL': 'NIFTY METAL',  # Metal index
    }
    
    # MFAPI base URL
    MFAPI_BASE_URL = "https://api.mfapi.in/mf"
    
    def __init__(self, period_years: int = 1, use_upstox: bool = False):
        """
        Initialize data fetcher
        
        Args:
            period_years: Number of years of historical data to fetch
            use_upstox: Use Upstox API for Indian stocks (True) or yfinance (False)
                       Note: Set to False by default as Upstox historical data may require subscription
        """
        self.period_years = period_years
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=period_years * 365 + 30)  # Add buffer
        self.mf_cache = {}  # Cache for mutual fund scheme codes
        self.use_upstox = use_upstox
        self.upstox_api = None
        
        # Initialize Upstox API if credentials are available
        if self.use_upstox and Config.validate_upstox_credentials():
            try:
                self._initialize_upstox()
                print("ℹ️  Upstox API initialized (historical data may require paid subscription)")
            except Exception as e:
                print(f"Warning: Failed to initialize Upstox API: {str(e)}")
                print("Falling back to yfinance for stock data")
                self.use_upstox = False
        elif self.use_upstox:
            print("Warning: Upstox credentials not configured")
            print("Falling back to yfinance for stock data")
            self.use_upstox = False
        else:
            print("ℹ️  Using yfinance for stock data (Upstox disabled)")
    
    def _initialize_upstox(self):
        """Initialize Upstox API client"""
        configuration = upstox_client.Configuration()
        configuration.access_token = Config.UPSTOX_ACCESS_TOKEN
        
        api_client = upstox_client.ApiClient(configuration)
        self.upstox_api = upstox_client.HistoryApi(api_client)
        self.upstox_market_api = upstox_client.MarketQuoteApi(api_client)
    
    def get_instrument_key(self, symbol: str, exchange: str = 'NSE_EQ') -> Optional[str]:
        """
        Get proper instrument key for Upstox by searching
        
        Args:
            symbol: Stock symbol
            exchange: Exchange identifier
            
        Returns:
            Proper instrument key or None
        """
        try:
            # For NSE stocks, the format is typically: NSE_EQ|INE...01 (ISIN) or NSE_EQ|SYMBOL
            # Try standard format first
            return f"{exchange}|{symbol}"
            
        except Exception as e:
            print(f"Error getting instrument key for {symbol}: {str(e)}")
            return None
    
    def normalize_ticker(self, instrument: str) -> tuple:
        """
        Convert instrument name to appropriate ticker symbol
        
        Args:
            instrument: Instrument name from portfolio
            
        Returns:
            Tuple of (ticker_symbol, exchange, use_upstox)
        """
        # Check if it's in our mappings
        if instrument.upper() in self.TICKER_MAPPINGS:
            ticker = self.TICKER_MAPPINGS[instrument.upper()]
            
            # Gold should use yfinance
            if instrument.upper() == 'GOLD1':
                return ('GC=F', None, False)
            
            # For stocks, use Upstox if available
            return (ticker, 'NSE_EQ', self.use_upstox)
        
        # For mutual funds, return None to signal MFAPI usage
        if 'fund' in instrument.lower():
            return (None, None, False)
        
        # Default: assume it's an NSE stock
        return (instrument.upper(), 'NSE_EQ', self.use_upstox)
    
    def fetch_upstox_data(self, symbol: str, exchange: str = 'NSE_EQ') -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Upstox API
        
        Args:
            symbol: Stock symbol (e.g., 'NATIONALUM')
            exchange: Exchange identifier (e.g., 'NSE_EQ')
            
        Returns:
            DataFrame with date index and close prices
        """
        try:
            if not self.upstox_api:
                return None
            
            # Create instrument key for Upstox
            instrument_key = f"{exchange}|{symbol}"
            
            # Convert dates to required format
            from_date = self.start_date.strftime('%Y-%m-%d')
            to_date = self.end_date.strftime('%Y-%m-%d')
            
            # Fetch historical data (daily interval)
            api_response = self.upstox_api.get_historical_candle_data1(
                instrument_key=instrument_key,
                interval='day',
                to_date=to_date,
                from_date=from_date,
                api_version='2.0'
            )
            
            if not api_response or not hasattr(api_response, 'data'):
                print(f"No data returned from Upstox for {symbol}")
                return None
            
            # Parse candle data
            candles = api_response.data.candles
            
            if not candles:
                print(f"No candle data for {symbol}")
                return None
            
            # Convert to DataFrame
            # Candle format: [timestamp, open, high, low, close, volume, oi]
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_localize(None)  # Make timezone-naive
            
            # Sort by date
            df = df.sort_index()
            
            # Return only close prices
            return df[['close']].rename(columns={'close': symbol})
            
        except ApiException as e:
            print(f"Upstox API error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"Error fetching Upstox data for {symbol}: {str(e)}")
            return None
    
    def fetch_price_data(self, ticker: str, exchange: str = None, use_upstox: bool = False) -> pd.DataFrame:
        """
        Fetch historical price data for a single ticker
        
        Args:
            ticker: Ticker symbol
            exchange: Exchange for Upstox (e.g., 'NSE_EQ')
            use_upstox: Whether to use Upstox API
            
        Returns:
            DataFrame with date index and adjusted close prices
        """
        # Try Upstox first if enabled
        if use_upstox and exchange and self.upstox_api:
            data = self.fetch_upstox_data(ticker, exchange)
            if data is not None:
                return data
            print(f"Upstox fetch failed for {ticker}, falling back to yfinance...")
        
        # Fallback to yfinance
        try:
            if ticker is None:
                return None
            
            # For yfinance, add .NS suffix if it's an Indian stock
            yf_ticker = ticker
            if exchange == 'NSE_EQ':
                yf_ticker = f"{ticker}.NS"
            
            stock = yf.Ticker(yf_ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                print(f"Warning: No data found for {yf_ticker}")
                return None
            
            # Convert to timezone-naive for consistency with MFAPI
            df.index = df.index.tz_localize(None)
            
            # Return only adjusted close prices
            return df[['Close']].rename(columns={'Close': ticker})
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def search_mutual_fund(self, fund_name: str) -> int:
        """
        Search for mutual fund scheme code using MFAPI
        
        Args:
            fund_name: Name of the mutual fund
            
        Returns:
            Scheme code (int) or None if not found
        """
        try:
            # Check cache first
            if fund_name in self.mf_cache:
                return self.mf_cache[fund_name]
            
            # Search for the fund
            search_url = f"{self.MFAPI_BASE_URL}/search"
            params = {'q': fund_name.split()[0:3]}  # Use first 3 words for search
            search_query = ' '.join(fund_name.split()[0:3])
            params = {'q': search_query}
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            
            if results and len(results) > 0:
                # Try to find best match - prefer Direct Plan Growth
                for result in results:
                    scheme_name = result.get('schemeName', '').lower()
                    if 'direct' in scheme_name and 'growth' in scheme_name:
                        if any(word.lower() in scheme_name for word in fund_name.split()[0:3]):
                            scheme_code = result.get('schemeCode')
                            self.mf_cache[fund_name] = scheme_code
                            return scheme_code
                
                # If no direct growth found, use first result
                scheme_code = results[0].get('schemeCode')
                self.mf_cache[fund_name] = scheme_code
                return scheme_code
            
            return None
            
        except Exception as e:
            print(f"Error searching mutual fund {fund_name}: {str(e)}")
            return None
    
    def fetch_mutual_fund_data(self, fund_name: str) -> pd.DataFrame:
        """
        Fetch historical NAV data for mutual fund from MFAPI
        
        Args:
            fund_name: Name of the mutual fund
            
        Returns:
            DataFrame with date index and NAV values
        """
        try:
            # Get scheme code
            scheme_code = self.search_mutual_fund(fund_name)
            
            if not scheme_code:
                print(f"Could not find scheme code for {fund_name}")
                return None
            
            # Fetch NAV history
            url = f"{self.MFAPI_BASE_URL}/{scheme_code}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'SUCCESS':
                print(f"API returned non-success status for {fund_name}")
                return None
            
            # Parse NAV data
            nav_data = data.get('data', [])
            
            if not nav_data:
                print(f"No NAV data available for {fund_name}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(nav_data)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'])
            df = df.sort_values('date')
            df.set_index('date', inplace=True)
            
            # Filter by date range
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            if df.empty:
                print(f"No data in date range for {fund_name}")
                return None
            
            # Rename column to match ticker format
            df = df[['nav']].rename(columns={'nav': fund_name})
            
            scheme_name = data.get('meta', {}).get('scheme_name', fund_name)
            print(f"Found: {scheme_name} (Code: {scheme_code})")
            
            return df
            
        except Exception as e:
            print(f"Error fetching mutual fund data for {fund_name}: {str(e)}")
            return None
    
    def fetch_portfolio_data(self, instruments: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all portfolio instruments
        
        Args:
            instruments: List of instrument names
            
        Returns:
            Dictionary mapping ticker to price DataFrame
        """
        portfolio_data = {}
        
        for instrument in instruments:
            # Check if it's a mutual fund
            if 'fund' in instrument.lower():
                print(f"Fetching mutual fund data for {instrument}...")
                data = self.fetch_mutual_fund_data(instrument)
                if data is not None:
                    portfolio_data[instrument] = data
                else:
                    print(f"Failed to fetch mutual fund data for {instrument}")
            else:
                ticker, exchange, use_api = self.normalize_ticker(instrument)
                
                if ticker:
                    source = "Upstox" if (use_api and self.upstox_api) else "yfinance"
                    print(f"Fetching data for {instrument} ({ticker}) via {source}...")
                    data = self.fetch_price_data(ticker, exchange, use_api)
                    
                    if data is not None:
                        # Rename column to original instrument name
                        data.columns = [instrument]
                        portfolio_data[instrument] = data
                else:
                    print(f"Skipping {instrument} - could not normalize ticker")
        
        return portfolio_data
    
    def fetch_benchmark_data(self, benchmark_ticker: str) -> pd.DataFrame:
        """
        Fetch benchmark index data
        
        Args:
            benchmark_ticker: Benchmark ticker symbol
            
        Returns:
            DataFrame with benchmark prices
        """
        print(f"Fetching benchmark data: {benchmark_ticker}")
        
        # For indices, use yfinance regardless of Upstox setting
        # Upstox indices require special handling
        return self.fetch_price_data(benchmark_ticker, exchange=None, use_upstox=False)
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns from price data
        
        Args:
            price_data: DataFrame with prices
            
        Returns:
            Series of daily returns
        """
        if price_data is None or price_data.empty:
            return None
        
        # Calculate percentage returns
        returns = price_data.pct_change().dropna()
        return returns.iloc[:, 0]  # Return first column as Series
    
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (Indian T-Bill rate)
        Using approximation of 6.5% annual (India govt bonds)
        
        Returns:
            Annual risk-free rate as decimal
        """
        # In production, fetch from RBI or reliable source
        # For now, using conservative estimate
        return 0.065
    
    def get_sector_benchmark(self, sector: str) -> str:
        """
        Map sector to appropriate benchmark index
        
        Args:
            sector: Sector name
            
        Returns:
            Benchmark ticker symbol
        """
        sector_benchmarks = {
            'Precious Metals': 'GC=F',  # Gold futures
            'Mining': '^CNXMETAL',  # Metal index
            'Metal': '^CNXMETAL',
            'Large and Mid Cap Fund': '^NSEI',  # Nifty 50
            'Technology': '^CNXIT',  # IT index
            'Banking': '^NSEBANK',  # Bank Nifty
            'Pharma': '^CNXPHARMA',
            'Auto': '^CNXAUTO',
        }
        
        return sector_benchmarks.get(sector, '^NSEI')  # Default to Nifty 50
