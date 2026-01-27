"""
Phase 4: Feature Engineering with Investor Feature Prioritization
Prepares training data for ML model, prioritizing investor profile features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering pipeline with investor feature prioritization
    
    Priority Tiers:
    1. Investor Features (6) - HIGHEST PRIORITY
    2. Asset Features (9)
    3. Market Context Features (11)
    4. Portfolio Features (9) - LOWEST (low variance)
    """
    
    # Feature groupings
    INVESTOR_FEATURES = [
        'investor_risk_capacity',
        'investor_risk_tolerance',
        'investor_behavioral_fragility',
        'investor_time_horizon_strength',
        'investor_effective_risk_tolerance',
        'investor_time_horizon_years'
    ]
    
    ASSET_FEATURES = [
        'asset_returns_60d_ma',
        'asset_volatility_30d',
        'asset_sharpe_ratio',
        'asset_sortino_ratio',
        'asset_calmar_ratio',
        'asset_max_drawdown',
        'asset_skewness',
        'asset_kurtosis',
        'asset_beta'
    ]
    
    MARKET_FEATURES = [
        'market_vix',
        'market_volatility_level',
        'market_vix_percentile',
        'nifty50_level',
        'market_return_1m',
        'market_return_3m',
        'market_regime_bull',
        'market_regime_bear',
        'risk_free_rate',
        'market_top_sector_return',
        'market_bottom_sector_return',
        'market_sector_return_dispersion'
    ]
    
    PORTFOLIO_FEATURES = [
        'portfolio_num_holdings',
        'portfolio_value',
        'portfolio_sector_concentration',
        'portfolio_equity_pct',
        'portfolio_commodity_pct',
        'portfolio_avg_weight',
        'portfolio_volatility',
        'portfolio_sharpe',
        'portfolio_max_drawdown'
    ]
    
    def __init__(self, data_path: str = "f:\\AI Insights Dashboard\\labeled_training_data.csv",
                 scaling_method: str = 'minmax'):
        """
        Initialize feature engineer
        
        Args:
            data_path: Path to labeled training data
            scaling_method: 'minmax' [0,1] or 'standard' (mean=0, std=1)
        """
        self.data_path = data_path
        self.scaling_method = scaling_method
        self.df = pd.read_csv(data_path)
        self.scaler = MinMaxScaler() if scaling_method == 'minmax' else StandardScaler()
        
        print(f"\n{'='*80}")
        print(f"PHASE 4: FEATURE ENGINEERING (INVESTOR-PRIORITIZED)")
        print(f"{'='*80}")
        print(f"\n📁 Loaded data: {data_path}")
        print(f"   Samples: {len(self.df)}")
        print(f"   Features: {len(self.df.columns) - 2}")  # -2 for stock_symbol and targets
    
    def analyze_feature_variance(self) -> dict:
        """
        Analyze variance by feature category to identify low-variance features
        
        Returns:
            Dictionary with variance statistics
        """
        print(f"\n{'─'*80}")
        print(f"STEP 1: ANALYZE FEATURE VARIANCE")
        print(f"{'─'*80}")
        
        variance_stats = {}
        
        for category, features in [
            ('Investor', self.INVESTOR_FEATURES),
            ('Asset', self.ASSET_FEATURES),
            ('Market', self.MARKET_FEATURES),
            ('Portfolio', self.PORTFOLIO_FEATURES)
        ]:
            variances = []
            for feature in features:
                if feature in self.df.columns:
                    var = self.df[feature].var()
                    variances.append(var)
            
            mean_var = np.mean(variances) if variances else 0
            variance_stats[category] = {
                'mean_variance': mean_var,
                'num_features': len(features),
                'features': features
            }
            
            print(f"\n{category} Features:")
            print(f"  Count: {len(features)}")
            print(f"  Mean Variance: {mean_var:.4f}")
            if category == 'Portfolio':
                print(f"  ⚠️  Low variance detected - consider these as context rather than predictors")
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Check and handle missing values
        
        Returns:
            DataFrame with missing values handled
        """
        print(f"\n{'─'*80}")
        print(f"STEP 2: HANDLE MISSING VALUES")
        print(f"{'─'*80}")
        
        missing_count = self.df.isnull().sum().sum()
        print(f"\nTotal missing values: {missing_count}")
        
        if missing_count > 0:
            missing_by_column = self.df.isnull().sum()
            print(f"\nMissing by column:")
            print(missing_by_column[missing_by_column > 0])
            
            # Forward fill then backward fill
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
            print(f"✓ Applied forward fill → backward fill")
        else:
            print(f"✓ No missing values found")
        
        return self.df
    
    def scale_features(self, features_to_scale: list = None) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            features_to_scale: List of features to scale (default: all numeric)
            
        Returns:
            DataFrame with scaled features
        """
        print(f"\n{'─'*80}")
        print(f"STEP 3: SCALE FEATURES ({self.scaling_method.upper()})")
        print(f"{'─'*80}")
        
        if features_to_scale is None:
            features_to_scale = (
                self.INVESTOR_FEATURES + 
                self.ASSET_FEATURES + 
                self.MARKET_FEATURES + 
                self.PORTFOLIO_FEATURES
            )
        
        # Filter to features that exist in dataframe
        features_to_scale = [f for f in features_to_scale if f in self.df.columns]
        
        # Scale features
        self.df[features_to_scale] = self.scaler.fit_transform(
            self.df[features_to_scale]
        )
        
        print(f"\n✓ Scaled {len(features_to_scale)} features using {self.scaling_method.upper()}")
        print(f"   Method: {'MinMaxScaler [0,1]' if self.scaling_method == 'minmax' else 'StandardScaler (mean=0, std=1)'}")
        
        return self.df
    
    def remove_correlated_features(self, correlation_threshold: float = 0.95) -> dict:
        """
        Identify and mark highly correlated features for removal
        
        Args:
            correlation_threshold: Remove features with correlation > threshold
            
        Returns:
            Dictionary with correlation analysis
        """
        print(f"\n{'─'*80}")
        print(f"STEP 4: CORRELATION ANALYSIS (Remove > {correlation_threshold})")
        print(f"{'─'*80}")
        
        # Calculate correlation matrix for numeric features
        numeric_features = self.INVESTOR_FEATURES + self.ASSET_FEATURES + self.MARKET_FEATURES + self.PORTFOLIO_FEATURES
        numeric_features = [f for f in numeric_features if f in self.df.columns]
        
        correlation_matrix = self.df[numeric_features].corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    high_correlations.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })
        
        print(f"\nFound {len(high_correlations)} highly correlated pairs (>{correlation_threshold}):")
        for pair in high_correlations:
            print(f"  {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}")
        
        if len(high_correlations) == 0:
            print(f"  ✓ No highly correlated features found - all features retained")
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_correlations
        }
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance using mutual information
        Prioritizes investor features
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            DataFrame with feature importance scores
        """
        print(f"\n{'─'*80}")
        print(f"STEP 5: FEATURE IMPORTANCE ANALYSIS (INVESTOR-PRIORITIZED)")
        print(f"{'─'*80}")
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores,
            'category': ['Investor' if f in self.INVESTOR_FEATURES 
                        else 'Asset' if f in self.ASSET_FEATURES
                        else 'Market' if f in self.MARKET_FEATURES
                        else 'Portfolio' 
                        for f in X.columns]
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 TOP 15 MOST IMPORTANT FEATURES:")
        print(f"\n{'Rank':<5} {'Feature':<40} {'Importance':<12} {'Category':<12}")
        print(f"{'-'*70}")
        for idx, row in importance_df.head(15).iterrows():
            print(f"{importance_df.index.get_loc(idx)+1:<5} {row['feature']:<40} {row['importance']:<12.4f} {row['category']:<12}")
        
        # Investor feature statistics
        investor_importance = importance_df[importance_df['category'] == 'Investor']
        print(f"\n👤 INVESTOR FEATURES IMPORTANCE:")
        print(f"   Average: {investor_importance['importance'].mean():.4f}")
        print(f"   Rank in Top 5: {len(investor_importance[investor_importance['importance'] >= importance_df.iloc[4]['importance']])}/6")
        
        return importance_df
    
    def create_train_test_split(self, test_size: float = 0.3, stratify: bool = True) -> dict:
        """
        Create stratified train/test split
        
        Args:
            test_size: Fraction for test set
            stratify: Use stratified split to maintain class ratio
            
        Returns:
            Dictionary with train/test splits
        """
        print(f"\n{'─'*80}")
        print(f"STEP 6: TRAIN/TEST SPLIT (Stratified={stratify})")
        print(f"{'─'*80}")
        
        # Prepare features and target
        feature_cols = (
            self.INVESTOR_FEATURES + 
            self.ASSET_FEATURES + 
            self.MARKET_FEATURES + 
            self.PORTFOLIO_FEATURES
        )
        feature_cols = [f for f in feature_cols if f in self.df.columns]
        
        X = self.df[feature_cols]
        y = self.df['recommendation_success']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y if stratify else None
        )
        
        print(f"\n✓ Train/Test Split Created:")
        print(f"   Training Set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"     ├─ Success: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        print(f"     └─ Failure: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
        print(f"\n   Test Set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"     ├─ Success: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")
        print(f"     └─ Failure: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%)")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_cols': feature_cols
        }
    
    def save_engineered_features(self, output_path: str = None) -> str:
        """
        Save engineered features for model training
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = "f:\\AI Insights Dashboard\\engineered_features.csv"
        
        self.df.to_csv(output_path, index=False)
        print(f"\n✅ Engineered features saved: {output_path}")
        print(f"   Size: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        return output_path


def main():
    """Run feature engineering pipeline"""
    
    # Initialize
    engineer = FeatureEngineer(scaling_method='minmax')
    
    # Step 1: Analyze variance
    engineer.analyze_feature_variance()
    
    # Step 2: Handle missing values
    engineer.handle_missing_values()
    
    # Step 3: Scale features
    engineer.scale_features()
    
    # Step 4: Correlation analysis
    correlation_analysis = engineer.remove_correlated_features(correlation_threshold=0.95)
    
    # Step 5: Feature importance
    feature_cols = (
        engineer.INVESTOR_FEATURES + 
        engineer.ASSET_FEATURES + 
        engineer.MARKET_FEATURES + 
        engineer.PORTFOLIO_FEATURES
    )
    feature_cols = [f for f in feature_cols if f in engineer.df.columns]
    
    importance_df = engineer.analyze_feature_importance(
        engineer.df[feature_cols],
        engineer.df['recommendation_success']
    )
    
    # Step 6: Train/test split
    split_data = engineer.create_train_test_split(test_size=0.3, stratify=True)
    
    # Save engineered features
    output_path = engineer.save_engineered_features()
    
    # Save train/test split for model training
    split_data['X_train'].to_csv("f:\\AI Insights Dashboard\\X_train.csv", index=False)
    split_data['X_test'].to_csv("f:\\AI Insights Dashboard\\X_test.csv", index=False)
    split_data['y_train'].to_csv("f:\\AI Insights Dashboard\\y_train.csv", index=False)
    split_data['y_test'].to_csv("f:\\AI Insights Dashboard\\y_test.csv", index=False)
    
    print(f"\n✅ Train/Test splits saved:")
    print(f"   X_train.csv, X_test.csv")
    print(f"   y_train.csv, y_test.csv")
    
    print(f"\n{'='*80}")
    print(f"✓ PHASE 4 FEATURE ENGINEERING COMPLETE")
    print(f"{'='*80}")
    print(f"\n📋 Summary:")
    print(f"   Features analyzed: {len(feature_cols)}")
    print(f"   Scaling method: MinMax [0,1]")
    print(f"   Priority: Investor Features")
    print(f"   Training samples: {len(split_data['X_train'])}")
    print(f"   Test samples: {len(split_data['X_test'])}")
    print(f"\nReady for Phase 5: ML Model Training (XGBoost)")


if __name__ == "__main__":
    main()
