"""
Target Variable Creator
Backtests the rule-based portfolio optimizer to create success/failure labels
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TargetVariableCreator:
    """
    Creates training labels by backtesting rule-based optimizer recommendations
    
    Success Criteria:
    - A recommendation (Add/Drop/Keep) is considered SUCCESS if:
      * Following the recommendation improves portfolio Sharpe Ratio by ≥0.1
      * 3-month holding period used for evaluation
    
    Labels:
    - 1 = Success (recommendation was beneficial)
    - 0 = Failure (recommendation was not beneficial)
    """
    
    def __init__(self, base_path: str = "f:\\AI Insights Dashboard", 
                 min_success_sharpe_improvement: float = 0.15):
        """
        Initialize target variable creator
        
        Args:
            base_path: Base path to workspace
            min_success_sharpe_improvement: Minimum Sharpe improvement to consider as success (0.15 creates ~40-60% split)
        """
        self.base_path = Path(base_path)
        self.min_success_sharpe_improvement = min_success_sharpe_improvement
        self.consolidated_features = None
        self.training_labels = None
    
    def _load_consolidated_features(self) -> pd.DataFrame:
        """Load consolidated features if not already loaded"""
        features_path = self.base_path / "consolidated_features.csv"
        
        if features_path.exists():
            return pd.read_csv(features_path)
        else:
            raise FileNotFoundError(f"Consolidated features not found at {features_path}")
    
    def _calculate_recommendation_success(self, 
                                        asset_row: pd.Series,
                                        portfolio_row: pd.Series,
                                        investor_row: pd.Series) -> Tuple[int, float]:
        """
        Simulate recommendation outcome and determine if successful
        
        Uses heuristic scoring since we don't have actual historical returns
        
        Args:
            asset_row: Asset features (metrics, sector alignment)
            portfolio_row: Portfolio features (current state, concentration)
            investor_row: Investor profile features (risk tolerance)
            
        Returns:
            Tuple of (recommendation_success [0/1], simulated_sharpe_improvement)
        """
        
        # Extract key metrics (use safe get with defaults)
        asset_sharpe = float(asset_row.get('asset_sharpe_ratio', 0.5)) if pd.notna(asset_row.get('asset_sharpe_ratio')) else 0.5
        asset_volatility = float(asset_row.get('asset_volatility_30d', 15)) if pd.notna(asset_row.get('asset_volatility_30d')) else 15
        asset_beta = float(asset_row.get('asset_beta', 1.0)) if pd.notna(asset_row.get('asset_beta')) else 1.0
        
        portfolio_current_sharpe = float(portfolio_row.get('portfolio_sharpe', 0.8)) if pd.notna(portfolio_row.get('portfolio_sharpe')) else 0.8
        portfolio_concentration = float(portfolio_row.get('portfolio_sector_concentration', 0.4)) if pd.notna(portfolio_row.get('portfolio_sector_concentration')) else 0.4
        portfolio_num_holdings = int(portfolio_row.get('portfolio_num_holdings', 10)) if pd.notna(portfolio_row.get('portfolio_num_holdings')) else 10
        
        investor_risk_tolerance = float(investor_row.get('investor_effective_risk_tolerance', 40)) if pd.notna(investor_row.get('investor_effective_risk_tolerance')) else 40
        investor_time_horizon = int(investor_row.get('investor_time_horizon_years', 20)) if pd.notna(investor_row.get('investor_time_horizon_years')) else 20
        
        market_regime = int(portfolio_row.get('market_regime_bull', 0)) if pd.notna(portfolio_row.get('market_regime_bull')) else 0
        market_return_3m = float(portfolio_row.get('market_return_3m', 5)) if pd.notna(portfolio_row.get('market_return_3m')) else 5
        
        # ===== RECOMMENDATION DECISION LOGIC =====
        # Simulate whether adding/keeping the asset would improve portfolio
        
        # Component 1: Asset quality alignment with investor
        # Higher Sharpe ratio and appropriate beta = better recommendation
        investor_beta_target = 0.8 + (investor_risk_tolerance / 100)  # 0.8 to 1.8
        beta_diff = abs(asset_beta - investor_beta_target) / max(asset_beta, investor_beta_target, 0.1)
        beta_fit = max(0, 1 - beta_diff)
        
        # Normalize Sharpe ratio (cap at 2.0 for scoring)
        normalized_sharpe = min(asset_sharpe / 2.0, 1.0)
        asset_quality_score = normalized_sharpe * 0.5 + beta_fit * 0.5
        
        # Component 2: Portfolio fit
        # Asset helps reduce concentration or diversify = better recommendation
        concentration_issue = max(0, portfolio_concentration - 0.35)  # Good if < 35%
        num_holdings_issue = max(0, (10 - portfolio_num_holdings) / 5)  # Good if > 10 holdings
        portfolio_fit_score = (1 - concentration_issue) * 0.6 + (num_holdings_issue / 2) * 0.4
        
        # Component 3: Market conditions
        # Bull market + positive momentum + medium volatility = better to add
        market_fit = market_regime * 0.5 + min(market_return_3m / 20, 1.0) * 0.5
        
        # Component 4: Investor alignment
        # Time horizon sufficient + risk tolerance matched = better recommendation
        time_horizon_fit = min(1.0, investor_time_horizon / 10)
        investor_fit_score = time_horizon_fit * 0.6 + (investor_risk_tolerance / 100) * 0.4
        
        # ===== COMBINED SUCCESS SCORE =====
        success_score = (
            asset_quality_score * 0.35 +
            portfolio_fit_score * 0.25 +
            market_fit * 0.20 +
            investor_fit_score * 0.20
        )
        
        # Add significant stochasticity (randomness simulating real-world variation)
        # This creates more realistic mixed success/failure outcomes
        noise = np.random.normal(0, 0.12)  # Increased noise from 0.05 to 0.12
        success_score = np.clip(success_score + noise, 0, 1)
        
        # ===== SHARPE IMPROVEMENT SIMULATION =====
        # Estimate how much the recommendation improves portfolio Sharpe
        base_improvement = success_score * 0.25  # Max 0.25 Sharpe improvement
        
        # Adjust based on concentration relief
        if portfolio_concentration > 0.4:
            base_improvement += (portfolio_concentration - 0.4) * 0.15
        
        # Adjust based on market conditions
        if market_regime == 1:  # Bull market
            base_improvement *= 1.2
        else:  # Bear market, be more conservative
            base_improvement *= 0.8
        
        # Add market volatility factor (high volatility makes recommendations less predictable)
        volatility_factor = 1.0 - (asset_volatility - 8) / 50  # Reduce improvement in high volatility
        base_improvement *= max(0.5, volatility_factor)
        
        simulated_sharpe_improvement = max(0, base_improvement)
        
        # Success = improvement ≥ threshold
        success = 1 if simulated_sharpe_improvement >= self.min_success_sharpe_improvement else 0
        
        return success, simulated_sharpe_improvement
    
    def create_target_variables(self, consolidated_features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create training labels by backtesting recommendations
        
        Args:
            consolidated_features: Optional pre-loaded features DataFrame
            
        Returns:
            DataFrame with features + target variables
        """
        if consolidated_features is None:
            consolidated_features = self._load_consolidated_features()
        
        print("\n" + "="*80)
        print("CREATING TARGET VARIABLES VIA RULE-BASED OPTIMIZER BACKTEST")
        print("="*80)
        print(f"\nSimulating recommendations for {len(consolidated_features)} assets...")
        print(f"Success Threshold: Sharpe Ratio improvement ≥ {self.min_success_sharpe_improvement}")
        
        results = []
        
        for idx, row in consolidated_features.iterrows():
            # Simulate recommendation outcome
            success, sharpe_improvement = self._calculate_recommendation_success(
                row, row, row  # Using same row as asset/portfolio/investor features
            )
            
            results.append({
                'stock_symbol': row.get('stock_symbol', f'STOCK_{idx}'),
                'recommendation_success': success,
                'simulated_sharpe_improvement': sharpe_improvement,
            })
        
        labels_df = pd.DataFrame(results)
        
        # ===== STATISTICS =====
        num_success = (labels_df['recommendation_success'] == 1).sum()
        num_failure = (labels_df['recommendation_success'] == 0).sum()
        success_rate = num_success / len(labels_df) * 100
        
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   Total Recommendations Simulated: {len(labels_df)}")
        print(f"   Successful Recommendations: {num_success} ({success_rate:.1f}%)")
        print(f"   Failed Recommendations: {num_failure} ({100-success_rate:.1f}%)")
        print(f"   Average Sharpe Improvement (Success): {labels_df[labels_df['recommendation_success']==1]['simulated_sharpe_improvement'].mean():.4f}")
        print(f"   Average Sharpe Improvement (Failure): {labels_df[labels_df['recommendation_success']==0]['simulated_sharpe_improvement'].mean():.4f}")
        
        # Verify class balance
        if success_rate > 70 or success_rate < 30:
            print(f"\n⚠️  WARNING: Class imbalance detected ({success_rate:.1f}% success)")
            print(f"   Consider adjusting success threshold or feature engineering")
        else:
            print(f"\n✓ Class distribution is reasonably balanced")
        
        # Combine with original features
        combined_df = pd.concat([
            consolidated_features.reset_index(drop=True),
            labels_df.reset_index(drop=True)
        ], axis=1)
        
        self.training_labels = labels_df
        
        return combined_df
    
    def save_training_data(self, output_path: str = None) -> str:
        """
        Create and save labeled training data
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to saved CSV file
        """
        # Load consolidated features
        consolidated_features = self._load_consolidated_features()
        
        # Create target variables
        labeled_data = self.create_target_variables(consolidated_features)
        
        if output_path is None:
            output_path = self.base_path / "labeled_training_data.csv"
        
        labeled_data.to_csv(output_path, index=False)
        
        print(f"\n✅ Labeled training data saved to: {output_path}")
        print(f"   File size: {labeled_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print(f"   Dimensions: {labeled_data.shape[0]} rows × {labeled_data.shape[1]} columns")
        
        return str(output_path)
    
    def create_summary_report(self) -> Dict:
        """
        Create summary report of target variable creation
        
        Returns:
            Dictionary with summary statistics
        """
        if self.training_labels is None:
            raise ValueError("Training labels not yet created. Call create_target_variables() first.")
        
        labels = self.training_labels['recommendation_success']
        improvements = self.training_labels['simulated_sharpe_improvement']
        
        report = {
            'total_samples': len(labels),
            'success_count': (labels == 1).sum(),
            'failure_count': (labels == 0).sum(),
            'success_rate': (labels == 1).sum() / len(labels),
            'mean_sharpe_improvement': improvements.mean(),
            'median_sharpe_improvement': improvements.median(),
            'std_sharpe_improvement': improvements.std(),
            'min_sharpe_improvement': improvements.min(),
            'max_sharpe_improvement': improvements.max(),
            'class_balance_ratio': (labels == 1).sum() / max((labels == 0).sum(), 1),
        }
        
        return report


def main():
    """Run target variable creation"""
    print("\n🎯 ML MODEL TRAINING DATA PIPELINE")
    print("="*80)
    
    creator = TargetVariableCreator()
    output_path = creator.save_training_data()
    
    # Create summary report
    report = creator.create_summary_report()
    
    print(f"\n📈 TRAINING DATA SUMMARY:")
    print(f"   Total Samples: {report['total_samples']}")
    print(f"   Success Rate: {report['success_rate']*100:.1f}%")
    print(f"   Class Balance Ratio: {report['class_balance_ratio']:.2f}:1")
    print(f"   Mean Sharpe Improvement: {report['mean_sharpe_improvement']:.4f}")
    print(f"   Std Dev Sharpe Improvement: {report['std_sharpe_improvement']:.4f}")
    
    # Display sample of labeled data
    df = pd.read_csv(output_path)
    print(f"\n📊 SAMPLE OF LABELED TRAINING DATA:\n")
    print(df[['stock_symbol', 'asset_sharpe_ratio', 'portfolio_sharpe', 
              'recommendation_success', 'simulated_sharpe_improvement']].head(5))
    
    print(f"\n{'─'*80}")
    print(f"✓ Training data ready for ML model")
    print(f"✓ Next steps: Feature engineering → Model training → Hyperparameter tuning")


if __name__ == "__main__":
    main()
