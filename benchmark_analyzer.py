"""
Benchmark Analyzer Module
Compares portfolio performance against sector-specific benchmarks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from data_fetcher import DataFetcher
from portfolio_metrics import PortfolioMetrics


class BenchmarkAnalyzer:
    """Analyze portfolio performance relative to benchmarks"""
    
    def __init__(self, data_fetcher: DataFetcher, metrics_calculator: PortfolioMetrics):
        """
        Initialize benchmark analyzer
        
        Args:
            data_fetcher: DataFetcher instance
            metrics_calculator: PortfolioMetrics instance
        """
        self.data_fetcher = data_fetcher
        self.metrics_calc = metrics_calculator
        self.benchmark_data = {}
    
    def determine_portfolio_benchmark(self, portfolio_df: pd.DataFrame) -> str:
        """
        Determine the most appropriate benchmark for the portfolio
        Based on asset allocation
        
        Args:
            portfolio_df: Portfolio DataFrame with sectors and values
            
        Returns:
            Benchmark ticker symbol
        """
        # Calculate sector weights
        total_value = portfolio_df['Cur. val'].sum()
        portfolio_df['weight'] = portfolio_df['Cur. val'] / total_value
        
        # Aggregate by asset class
        asset_weights = portfolio_df.groupby('Asset Class')['weight'].sum()
        
        # If majority equity, use Nifty 50
        if 'Equity' in asset_weights and asset_weights['Equity'] > 0.6:
            return '^NSEI'  # Nifty 50
        
        # If majority commodity, use commodity index
        elif 'Commodity' in asset_weights and asset_weights['Commodity'] > 0.5:
            return 'GC=F'  # Gold
        
        # Default to balanced benchmark
        else:
            return '^NSEI'
    
    def fetch_sector_benchmarks(self, sectors: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch benchmark data for all sectors in portfolio
        
        Args:
            sectors: List of unique sectors
            
        Returns:
            Dictionary mapping sector to benchmark data
        """
        sector_benchmarks = {}
        
        for sector in sectors:
            benchmark_ticker = self.data_fetcher.get_sector_benchmark(sector)
            
            if benchmark_ticker not in self.benchmark_data:
                data = self.data_fetcher.fetch_benchmark_data(benchmark_ticker)
                self.benchmark_data[benchmark_ticker] = data
            
            sector_benchmarks[sector] = self.benchmark_data[benchmark_ticker]
        
        return sector_benchmarks
    
    def compare_to_benchmark(self, 
                            portfolio_returns: pd.Series, 
                            benchmark_ticker: str) -> Dict:
        """
        Compare portfolio performance to benchmark
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_ticker: Benchmark ticker symbol
            
        Returns:
            Dictionary with comparison metrics
        """
        # Fetch benchmark data if not already loaded
        if benchmark_ticker not in self.benchmark_data:
            self.benchmark_data[benchmark_ticker] = self.data_fetcher.fetch_benchmark_data(benchmark_ticker)
        
        benchmark_data = self.benchmark_data[benchmark_ticker]
        
        if benchmark_data is None:
            return {
                'benchmark_name': benchmark_ticker,
                'comparison': 'Unable to fetch benchmark data',
                'outperformance': None
            }
        
        # Calculate benchmark returns
        benchmark_returns = self.data_fetcher.calculate_returns(benchmark_data)
        
        # Calculate metrics for both
        portfolio_metrics = self.metrics_calc.calculate_all_metrics(portfolio_returns, benchmark_returns)
        benchmark_metrics = self.metrics_calc.calculate_all_metrics(benchmark_returns)
        
        # Calculate outperformance
        outperformance = None
        if portfolio_metrics['annual_return'] and benchmark_metrics['annual_return']:
            outperformance = portfolio_metrics['annual_return'] - benchmark_metrics['annual_return']
        
        return {
            'benchmark_name': benchmark_ticker,
            'benchmark_return': benchmark_metrics['annual_return'],
            'portfolio_return': portfolio_metrics['annual_return'],
            'outperformance': outperformance,
            'portfolio_sharpe': portfolio_metrics['sharpe_ratio'],
            'benchmark_sharpe': benchmark_metrics['sharpe_ratio'],
            'jensens_alpha': portfolio_metrics.get('jensens_alpha'),
            'beta': portfolio_metrics.get('beta'),
            'beats_benchmark': outperformance > 0 if outperformance else None
        }
    
    def analyze_sector_performance(self, 
                                   portfolio_df: pd.DataFrame,
                                   holdings_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze each holding against its sector benchmark
        
        Args:
            portfolio_df: Portfolio DataFrame
            holdings_data: Dictionary of holdings price data
            
        Returns:
            Dictionary with sector-level analysis
        """
        sector_analysis = {}
        
        for idx, row in portfolio_df.iterrows():
            instrument = row['Instrument']
            sector = row['Sector']
            
            # Get holding data
            if instrument not in holdings_data or holdings_data[instrument] is None:
                continue
            
            # Calculate returns
            returns = self.data_fetcher.calculate_returns(holdings_data[instrument])
            
            # Get sector benchmark
            benchmark_ticker = self.data_fetcher.get_sector_benchmark(sector)
            
            # Compare
            comparison = self.compare_to_benchmark(returns, benchmark_ticker)
            
            sector_analysis[instrument] = {
                'sector': sector,
                'benchmark': benchmark_ticker,
                'comparison': comparison
            }
        
        return sector_analysis
    
    def generate_benchmark_summary(self, 
                                   portfolio_returns: pd.Series,
                                   portfolio_df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive benchmark comparison summary
        
        Args:
            portfolio_returns: Aggregated portfolio returns
            portfolio_df: Portfolio DataFrame
            
        Returns:
            Summary dictionary
        """
        # Determine primary benchmark
        primary_benchmark = self.determine_portfolio_benchmark(portfolio_df)
        
        # Compare to primary benchmark
        primary_comparison = self.compare_to_benchmark(portfolio_returns, primary_benchmark)
        
        # Also compare to Nifty 50 if not already primary
        nifty_comparison = None
        if primary_benchmark != '^NSEI':
            nifty_comparison = self.compare_to_benchmark(portfolio_returns, '^NSEI')
        
        return {
            'primary_benchmark': primary_comparison,
            'nifty_50_comparison': nifty_comparison,
            'summary': self._generate_text_summary(primary_comparison)
        }
    
    def _generate_text_summary(self, comparison: Dict) -> str:
        """Generate human-readable summary"""
        if comparison['outperformance'] is None:
            return "Unable to compare due to missing data"
        
        if comparison['beats_benchmark']:
            return (f"Portfolio OUTPERFORMED {comparison['benchmark_name']} by "
                   f"{comparison['outperformance']:.2f}% annually")
        else:
            return (f"Portfolio UNDERPERFORMED {comparison['benchmark_name']} by "
                   f"{abs(comparison['outperformance']):.2f}% annually")
