"""
Portfolio Optimizer with Asset Universe Consideration
Recommends assets to add or drop from the portfolio to align with investor goals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from portfolio_metrics import PortfolioMetrics
from portfolio_quality import PortfolioQualityScorer
from investor_fit import InvestorFitScorer
from objective_alignment import ObjectiveAlignmentAnalyzer, InvestmentObjective


class PortfolioOptimizer:
    """Optimize portfolio by recommending assets to add/drop from available universe"""
    
    # Nifty50 stocks
    NIFTY50 = [
        'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV',
        'BPCL', 'BHARTIARTL', 'BOSCHIND', 'BRITANNIA', 'CIPLA',
        'COALINDIA', 'COLPAL', 'DIVISLAB', 'DRREDDY', 'EICHERMOT',
        'GAIL', 'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK',
        'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'HDFC',
        'ICICIBANK', 'IPCALAB', 'INDIGO', 'INFY', 'IOTACOMM',
        'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'LTTS',
        'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC',
        'POWERGRID', 'RELIANCE', 'SBICARD', 'SBILIFE', 'SBIN',
        'SUNPHARMA', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'TCS',
        'TECHM', 'TITAN', 'UPL', 'VBL', 'WIPRO'
    ]
    
    # Nifty Next50 stocks
    NIFTY_NEXT50 = [
        'ABB', 'AARTIIND', 'ABCAPITAL', 'ABSL', 'ADANIENT',
        'ADANIGREEN', 'ADANIPOWER', 'AGNC', 'AGRTECH', 'AMARAJABAT',
        'AMBUJACEM', 'APOLLOHOSP', 'APOLLOTYRE', 'ASHOKLEY', 'AUROPHARMA',
        'AUBANK', 'AUPHARMACY', 'AUTOCLAD', 'AUTOIND', 'AWHCL',
        'AXISGO', 'AXISMF', 'BAJAJTRIPUR', 'BARBEQUE', 'BASF',
        'BATAINDIA', 'BEL', 'BERGEPAINT', 'BHARATRAS', 'BHEL',
        'BIGBLOC', 'BLS', 'BLUEBLITZ', 'BMS', 'BONDADV',
        'BRIGADE', 'CADILAHC', 'CAMS', 'CARBORUNM', 'CASTROL',
        'CCL', 'CDSL', 'CGCONSTRUCT', 'CGPOWER', 'CHAMBLFERT',
        'CHEMPLAST', 'CHOLAFIN', 'CHROMIND', 'CLEAN', 'COALGO'
    ]
    
    def __init__(self, current_holdings: Dict[str, float], investor_profile: Dict = None,
                 investment_objective: str = "Moderate Growth", data_fetcher: DataFetcher = None):
        """
        Initialize portfolio optimizer
        
        Args:
            current_holdings: Dict of {symbol: weight} for current portfolio
            investor_profile: Investor profile from IID
            investment_objective: Investment objective (Conservative/Moderate/Aggressive/Balanced)
            data_fetcher: Optional DataFetcher instance for price data
        """
        self.current_holdings = current_holdings
        self.current_symbols = set(current_holdings.keys())
        self.investor_profile = investor_profile or {}
        self.investment_objective = investment_objective
        self.data_fetcher = data_fetcher or DataFetcher(period_years=1)
        
        # Initialize scorers and analyzers
        self.quality_scorer = PortfolioQualityScorer()
        self.fit_scorer = InvestorFitScorer()
        self.objective_analyzer = ObjectiveAlignmentAnalyzer()
        self.metrics_calc = PortfolioMetrics(risk_free_rate=self.data_fetcher.get_risk_free_rate())
        
        # Asset universe
        self.asset_universe = list(set(self.NIFTY50 + self.NIFTY_NEXT50))
        
        # Available assets to consider (universe minus current holdings)
        self.available_assets = [s for s in self.asset_universe if s not in self.current_symbols]
        
        # Cache for asset metrics
        self.asset_metrics_cache = {}
        
    def get_asset_metrics(self, symbol: str) -> Dict:
        """
        Get individual asset metrics (return, volatility, sharpe, etc.)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with asset metrics
        """
        if symbol in self.asset_metrics_cache:
            return self.asset_metrics_cache[symbol]
        
        try:
            # Fetch price data
            price_data = self.data_fetcher.fetch_price_data(symbol, exchange='NSE_EQ')
            
            if price_data is None or len(price_data) < 20:
                # Use dummy metrics if data unavailable
                return {
                    'symbol': symbol,
                    'returns': 0.0,
                    'volatility': 0.25,
                    'sharpe': 0.0,
                    'sortino': 0.0,
                    'max_drawdown': 50.0,
                    'data_available': False
                }
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            ann_return = float(returns.mean() * 252)
            ann_vol = float(returns.std() * np.sqrt(252))
            
            # Sharpe and Sortino
            risk_free_rate = self.data_fetcher.get_risk_free_rate()
            sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0.0 else 0.0
            
            # Sortino (only downside volatility)
            downside_returns = returns[returns < 0]
            downside_vol = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else ann_vol
            sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0.0 else 0.0
            
            # Max drawdown
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.cummax()
            drawdown = (cum_returns - peak) / peak
            max_dd = float(abs(drawdown.min()) * 100)
            
            metrics = {
                'symbol': symbol,
                'returns': ann_return * 100,
                'volatility': ann_vol * 100,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_dd,
                'data_available': True
            }
            
            self.asset_metrics_cache[symbol] = metrics
            return metrics
            
        except Exception as e:
            print(f"Error getting metrics for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'returns': 0.0,
                'volatility': 25.0,
                'sharpe': 0.0,
                'sortino': 0.0,
                'max_drawdown': 50.0,
                'data_available': False
            }
    
    def score_asset_for_portfolio(self, symbol: str, current_portfolio_df: pd.DataFrame = None) -> Dict:
        """
        Score how good an asset would be for the portfolio
        
        Args:
            symbol: Stock symbol to score
            current_portfolio_df: Current portfolio returns DataFrame
            
        Returns:
            Dictionary with asset score and rationale
        """
        metrics = self.get_asset_metrics(symbol)
        
        # Get objective criteria
        objective_enum = self.objective_analyzer.parse_objective(self.investment_objective)
        criteria = self.objective_analyzer.OBJECTIVE_CRITERIA[objective_enum]
        
        score = 0
        rationale = []
        
        # Alignment with objective thresholds
        if criteria['min_return'] <= metrics['returns'] <= criteria['max_return']:
            score += 25
            rationale.append(f"✓ Return {metrics['returns']:.1f}% aligns with {objective_enum.value}")
        else:
            rationale.append(f"✗ Return {metrics['returns']:.1f}% outside target range")
        
        if metrics['volatility'] <= criteria['max_volatility']:
            score += 20
            rationale.append(f"✓ Volatility {metrics['volatility']:.1f}% is acceptable")
        else:
            rationale.append(f"✗ Volatility {metrics['volatility']:.1f}% exceeds limit")
        
        if metrics['sharpe'] >= criteria['min_sharpe']:
            score += 20
            rationale.append(f"✓ Sharpe ratio {metrics['sharpe']:.2f} is good")
        else:
            rationale.append(f"✗ Sharpe ratio {metrics['sharpe']:.2f} is weak")
        
        if metrics['max_drawdown'] <= 50:  # Reasonable drawdown limit
            score += 15
            rationale.append(f"✓ Max drawdown {metrics['max_drawdown']:.1f}% is manageable")
        else:
            rationale.append(f"✗ Max drawdown {metrics['max_drawdown']:.1f}% is high")
        
        # Diversification bonus - assets with low correlation to current holdings
        if current_portfolio_df is not None:
            try:
                asset_returns = self.data_fetcher.fetch_price_data(symbol, exchange='NSE_EQ')
                if asset_returns is not None and len(asset_returns) > 0:
                    asset_returns = asset_returns.pct_change().dropna()
                    
                    # Align dates with current portfolio
                    common_dates = current_portfolio_df.index.intersection(asset_returns.index)
                    if len(common_dates) > 20:
                        corr = current_portfolio_df.loc[common_dates].corrwith(
                            asset_returns.loc[common_dates]
                        ).mean()
                        
                        if corr < 0.5:  # Low correlation
                            score += 20
                            rationale.append(f"✓ Low correlation ({corr:.2f}) provides diversification")
                        else:
                            rationale.append(f"⚠ High correlation ({corr:.2f}) with current holdings")
            except:
                pass
        
        return {
            'symbol': symbol,
            'score': min(100, score),
            'metrics': metrics,
            'rationale': rationale,
            'recommendation_strength': 'Strong' if score >= 75 else 'Moderate' if score >= 50 else 'Weak'
        }
    
    def identify_poor_performers(self, portfolio_returns: pd.DataFrame, 
                                current_holdings_df: pd.DataFrame) -> List[Dict]:
        """
        Identify underperforming assets in current portfolio to consider dropping
        
        Args:
            portfolio_returns: Returns of current portfolio
            current_holdings_df: Current holdings data
            
        Returns:
            List of assets with poor performance scores
        """
        poor_performers = []
        
        # Check if portfolio_returns is empty or invalid
        if portfolio_returns is None or len(portfolio_returns) == 0:
            return poor_performers
        
        for symbol in self.current_symbols:
            try:
                # Get asset price data
                price_data = self.data_fetcher.fetch_price_data(symbol, exchange='NSE_EQ')
                
                if price_data is None or len(price_data) < 20:
                    continue
                
                returns = price_data.pct_change().dropna()
                
                # Align with portfolio period
                if isinstance(portfolio_returns, pd.Series):
                    common_dates = portfolio_returns.index.intersection(returns.index)
                else:
                    common_dates = []
                
                if len(common_dates) < 20:
                    continue
                
                # Compare with portfolio
                asset_ret = float(returns.loc[common_dates].mean() * 252)
                asset_vol = float(returns.loc[common_dates].std() * np.sqrt(252))
                portfolio_ret = float(portfolio_returns.loc[common_dates].mean() * 252)
                portfolio_vol = float(portfolio_returns.loc[common_dates].std() * np.sqrt(252))
                
                # Score: lower is worse
                if asset_vol > 0.0 and portfolio_vol > 0.0:
                    asset_sharpe = asset_ret / asset_vol
                    portfolio_sharpe = portfolio_ret / portfolio_vol
                    
                    # If asset underperforms portfolio significantly
                    underperformance = portfolio_sharpe - asset_sharpe
                    
                    if underperformance > 0.3:  # Significant underperformance
                        # Get weight
                        weight = self.current_holdings.get(symbol, 0)
                        
                        poor_performers.append({
                            'symbol': symbol,
                            'weight': weight * 100,
                            'asset_return': asset_ret * 100,
                            'asset_volatility': asset_vol * 100,
                            'asset_sharpe': asset_sharpe,
                            'portfolio_sharpe': portfolio_sharpe,
                            'underperformance': underperformance,
                            'recommendation': 'Consider reducing or removing' if weight < 0.05 else 'Consider rebalancing'
                        })
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by underperformance
        poor_performers.sort(key=lambda x: x['underperformance'], reverse=True)
        return poor_performers
    
    def generate_recommendations(self, portfolio_returns: pd.DataFrame,
                                current_holdings_df: pd.DataFrame = None,
                                max_additions: int = 5,
                                max_removals: int = 3) -> Dict:
        """
        Generate comprehensive asset recommendations
        
        Args:
            portfolio_returns: DataFrame of portfolio returns
            current_holdings_df: Current holdings data
            max_additions: Maximum number of assets to recommend adding
            max_removals: Maximum number of assets to recommend removing
            
        Returns:
            Dictionary with recommendations and rationale
        """
        print("=" * 80)
        print("GENERATING ASSET RECOMMENDATIONS")
        print("=" * 80)
        
        # Step 1: Identify poor performers to potentially drop
        print("\n🔍 Analyzing current holdings for underperformance...")
        poor_performers = []
        
        if isinstance(portfolio_returns, pd.Series) and len(portfolio_returns) > 0:
            poor_performers = self.identify_poor_performers(portfolio_returns, current_holdings_df)
        
        assets_to_drop = [
            p for p in poor_performers 
            if p['underperformance'] > 0.3
        ][:max_removals]
        
        print(f"✓ Found {len(assets_to_drop)} assets underperforming significantly")
        
        # Step 2: Score available assets for addition (limited to top 50 for speed)
        print("\n🔍 Scoring available assets from universe (limiting to 50 for speed)...")
        available_scores = []
        
        # Limit assets scored for speed
        assets_to_score = self.available_assets[:50]
        
        for i, symbol in enumerate(assets_to_score, 1):
            try:
                score_result = self.score_asset_for_portfolio(
                    symbol, 
                    current_holdings_df if current_holdings_df is not None else portfolio_returns
                )
                available_scores.append(score_result)
                if i % 10 == 0:
                    print(f"  Scored {i}/{len(assets_to_score)} assets...")
            except Exception as e:
                print(f"  ⚠️ Error scoring {symbol}: {str(e)}")
                continue
        
        # Sort by score
        available_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top candidates for addition
        assets_to_add = available_scores[:max_additions]
        
        print(f"✓ Top {len(assets_to_add)} candidates identified")
        
        # Step 3: Compile recommendations
        recommendations = {
            'objective': self.investment_objective,
            'assets_to_add': assets_to_add,
            'assets_to_drop': assets_to_drop,
            'rationale': {
                'why_add': f"These assets align better with {self.investment_objective} objective and provide diversification benefits",
                'why_drop': "These assets are underperforming relative to the portfolio and may be replaced with better alternatives"
            },
            'implementation_strategy': self._generate_implementation_strategy(
                assets_to_add, 
                assets_to_drop,
                current_holdings_df
            ),
            'expected_impact': self._calculate_expected_impact(
                assets_to_add,
                assets_to_drop
            )
        }
        
        return recommendations
    
    def _generate_implementation_strategy(self, assets_to_add: List[Dict], 
                                         assets_to_drop: List[Dict],
                                         current_holdings_df: pd.DataFrame = None) -> Dict:
        """Generate step-by-step implementation strategy"""
        
        total_drop_weight = sum(a['weight'] for a in assets_to_drop) if assets_to_drop else 0
        add_count = len(assets_to_add)
        
        strategy = {
            'phase_1': {
                'title': 'Phase 1: Reduce Underperformers (Weeks 1-2)',
                'actions': [
                    f"Sell {a['symbol']} (current weight: {a['weight']:.1f}%)" 
                    for a in assets_to_drop
                ],
                'freed_capital': total_drop_weight
            },
            'phase_2': {
                'title': 'Phase 2: Gradual Addition (Weeks 3-8)',
                'actions': [
                    f"Add {a['symbol']} (score: {a['score']:.0f}/100) with {total_drop_weight/add_count:.1f}% allocation"
                    for a in assets_to_add
                ] if add_count > 0 else [],
                'total_additions': add_count
            },
            'phase_3': {
                'title': 'Phase 3: Monitor & Rebalance (Ongoing)',
                'actions': [
                    'Monitor new holdings for 4 weeks',
                    'Rebalance if any weight drifts >5% from target',
                    'Track performance vs portfolio metrics'
                ]
            }
        }
        
        return strategy
    
    def _calculate_expected_impact(self, assets_to_add: List[Dict],
                                   assets_to_drop: List[Dict]) -> Dict:
        """
        Calculate expected impact on portfolio metrics.
        Focus on risk-adjusted returns (Sharpe ratio) rather than raw returns.
        """
        
        if not assets_to_add:
            return {
                'risk_adjusted_improvement': 0,
                'volatility_reduction': 0,
                'overall_impact_direction': 'Neutral',
                'interpretation': 'No changes recommended'
            }
        
        # Extract metrics from assets
        def get_returns(asset):
            if 'metrics' in asset:
                return asset['metrics'].get('returns', 0)
            else:
                return asset.get('asset_return', 0)
        
        def get_volatility(asset):
            if 'metrics' in asset:
                return asset['metrics'].get('volatility', 0)
            else:
                return asset.get('asset_volatility', 0)
        
        def get_sharpe(asset):
            if 'metrics' in asset:
                return asset['metrics'].get('sharpe', 0)
            else:
                return asset.get('asset_sharpe', 0)
        
        # Calculate average metrics for assets to add
        add_avg_return = np.mean([get_returns(a) for a in assets_to_add]) if assets_to_add else 0
        add_avg_vol = np.mean([get_volatility(a) for a in assets_to_add]) if assets_to_add else 0
        add_avg_sharpe = np.mean([get_sharpe(a) for a in assets_to_add]) if assets_to_add else 0
        
        # Calculate average metrics for assets to drop
        drop_avg_return = np.mean([get_returns(a) for a in assets_to_drop]) if assets_to_drop else 0
        drop_avg_vol = np.mean([get_volatility(a) for a in assets_to_drop]) if assets_to_drop else 0
        drop_avg_sharpe = np.mean([get_sharpe(a) for a in assets_to_drop]) if assets_to_drop else 0
        
        # Calculate improvements
        return_improvement = add_avg_return - drop_avg_return
        volatility_reduction = drop_avg_vol - add_avg_vol  # Positive if volatility decreases
        sharpe_improvement = add_avg_sharpe - drop_avg_sharpe  # Risk-adjusted return improvement
        
        # Determine overall impact
        if sharpe_improvement > 0.3:  # Significant risk-adjusted improvement
            direction = 'Positive'
            interpretation = 'Better risk-adjusted returns with improved Sharpe ratio'
        elif volatility_reduction > 2 and sharpe_improvement > 0:
            direction = 'Positive'
            interpretation = 'Lower volatility with maintained or improved returns (Better risk management)'
        elif sharpe_improvement > 0:
            direction = 'Positive'
            interpretation = 'Improved risk-adjusted returns despite potentially lower absolute returns'
        elif volatility_reduction > 0:
            direction = 'Mixed'
            interpretation = 'Lower volatility but reduced returns (Conservative shift)'
        else:
            direction = 'Negative'
            interpretation = 'No clear benefit to this change'
        
        return {
            'raw_return_change': return_improvement,
            'volatility_change': volatility_reduction,
            'sharpe_ratio_improvement': sharpe_improvement,
            'overall_impact_direction': direction,
            'interpretation': interpretation,
            'confidence': 'Medium',
            'details': {
                'assets_to_add_sharpe': round(add_avg_sharpe, 2),
                'assets_to_drop_sharpe': round(drop_avg_sharpe, 2),
                'assets_to_add_volatility': round(add_avg_vol, 2),
                'assets_to_drop_volatility': round(drop_avg_vol, 2)
            }
        }


def create_optimizer_from_analysis_result(analysis_result: Dict) -> Optional[PortfolioOptimizer]:
    """
    Create optimizer from portfolio analysis result
    
    Args:
        analysis_result: Result from PortfolioAnalyzer.run_complete_analysis()
        
    Returns:
        Initialized PortfolioOptimizer instance
    """
    try:
        # Extract current holdings
        portfolio_df = analysis_result.get('portfolio_df', pd.DataFrame())
        if portfolio_df.empty:
            return None
        
        # Calculate weights
        total_value = portfolio_df['Cur. val'].sum()
        current_holdings = {}
        
        for _, row in portfolio_df.iterrows():
            symbol = row['Symbol']
            weight = row['Cur. val'] / total_value if total_value > 0 else 0
            current_holdings[symbol] = weight
        
        # Extract investor profile
        investor_profile = analysis_result.get('investor_profile', {})
        
        # Get objective from analysis
        objective = analysis_result.get('investment_objective', 'Moderate Growth')
        
        return PortfolioOptimizer(
            current_holdings=current_holdings,
            investor_profile=investor_profile,
            investment_objective=objective
        )
        
    except Exception as e:
        print(f"Error creating optimizer: {str(e)}")
        return None
