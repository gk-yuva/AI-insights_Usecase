"""
Portfolio Metrics Calculator
Calculates risk and return metrics for portfolio analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class PortfolioMetrics:
    """Calculate various portfolio performance and risk metrics"""
    
    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default 6.5% for India)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1  # Convert to daily
    
    def calculate_annual_return(self, returns: pd.Series) -> float:
        """
        Calculate annualized return
        
        Args:
            returns: Series of daily returns (as decimals, e.g., 0.01 = 1%)
            
        Returns:
            Annualized return as percentage
        """
        if returns is None or len(returns) == 0:
            return None
        
        # Ensure returns are in decimal format (not percentages)
        # If returns appear to be percentages (> 1), convert to decimals
        if returns.mean() > 1:
            returns = returns / 100
        
        # Compound returns and annualize
        cumulative_return = (1 + returns).prod() - 1
        n_days = len(returns)
        annual_return = (1 + cumulative_return) ** (252 / n_days) - 1
        
        return annual_return * 100  # Return as percentage
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical method
        
        Args:
            returns: Series of daily returns
            confidence_level: Confidence level (0.95 for 95%, 0.99 for 99%)
            
        Returns:
            VaR as percentage (annualized)
        """
        if returns is None or len(returns) == 0:
            return None
        
        # Historical VaR: percentile of return distribution
        var_daily = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Annualize VaR (approximation using sqrt of time)
        var_annual = var_daily * np.sqrt(252)
        
        return abs(var_annual) * 100  # Return as positive percentage
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Args:
            returns: Series of daily returns
            confidence_level: Confidence level
            
        Returns:
            CVaR as percentage (annualized)
        """
        if returns is None or len(returns) == 0:
            return None
        
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        # Average of all returns worse than VaR
        cvar_daily = returns[returns <= var_threshold].mean()
        
        # Annualize
        cvar_annual = cvar_daily * np.sqrt(252)
        
        return abs(cvar_annual) * 100
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Annualized Sharpe Ratio
        """
        if returns is None or len(returns) == 0:
            return None
        
        # Calculate excess returns
        excess_returns = returns - self.daily_rf_rate
        
        # Annualized Sharpe
        if excess_returns.std() == 0:
            return 0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino Ratio (only penalizes downside volatility)
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Annualized Sortino Ratio
        """
        if returns is None or len(returns) == 0:
            return None
        
        # Calculate excess returns
        excess_returns = returns - self.daily_rf_rate
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        # Annualized Sortino
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        
        return sortino
    
    def calculate_jensens_alpha(self, 
                                portfolio_returns: pd.Series, 
                                benchmark_returns: pd.Series) -> float:
        """
        Calculate Jensen's Alpha (CAPM alpha)
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            
        Returns:
            Annualized Jensen's Alpha as percentage
        """
        if portfolio_returns is None or benchmark_returns is None:
            return None
        
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return None
        
        # Align the series
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 10:  # Need minimum data points
            return None
        
        port_excess = aligned_data['portfolio'] - self.daily_rf_rate
        bench_excess = aligned_data['benchmark'] - self.daily_rf_rate
        
        # Calculate beta using regression
        beta, alpha_daily, r_value, p_value, std_err = stats.linregress(
            bench_excess, port_excess
        )
        
        # Annualize alpha
        alpha_annual = alpha_daily * 252
        
        return alpha_annual * 100  # Return as percentage
    
    def calculate_beta(self, 
                       portfolio_returns: pd.Series, 
                       benchmark_returns: pd.Series) -> float:
        """
        Calculate portfolio beta relative to benchmark
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            
        Returns:
            Beta coefficient
        """
        if portfolio_returns is None or benchmark_returns is None:
            return None
        
        # Align the series
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 10:
            return None
        
        # Calculate covariance and variance
        covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
        benchmark_variance = np.var(aligned_data['benchmark'])
        
        if benchmark_variance == 0:
            return None
        
        beta = covariance / benchmark_variance
        
        return beta
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Maximum drawdown as percentage
        """
        if returns is None or len(returns) == 0:
            return None
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        
        return abs(max_dd) * 100  # Return as positive percentage
    
    def calculate_all_metrics(self, 
                             returns: pd.Series, 
                             benchmark_returns: pd.Series = None) -> Dict:
        """
        Calculate all portfolio metrics
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            Dictionary of all calculated metrics
        """
        metrics = {
            'annual_return': self.calculate_annual_return(returns),
            'var_95': self.calculate_var(returns, 0.95),
            'var_99': self.calculate_var(returns, 0.99),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'volatility': returns.std() * np.sqrt(252) * 100 if returns is not None else None
        }
        
        # Add benchmark-relative metrics if benchmark provided
        if benchmark_returns is not None:
            metrics['jensens_alpha'] = self.calculate_jensens_alpha(returns, benchmark_returns)
            metrics['beta'] = self.calculate_beta(returns, benchmark_returns)
        
        return metrics
