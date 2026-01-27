"""
Phase 5: ML Model Training with XGBoost
Trains classification model to predict recommendation success with investor feature prioritization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, auc
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class XGBoostModelTrainer:
    """
    Train and evaluate XGBoost classification model
    Predicts recommendation success (binary: 0/1)
    """
    
    def __init__(self, 
                 X_train_path: str = "f:\\AI Insights Dashboard\\X_train.csv",
                 y_train_path: str = "f:\\AI Insights Dashboard\\y_train.csv",
                 X_test_path: str = "f:\\AI Insights Dashboard\\X_test.csv",
                 y_test_path: str = "f:\\AI Insights Dashboard\\y_test.csv"):
        """
        Initialize trainer with data paths
        
        Args:
            X_train_path: Path to training features
            y_train_path: Path to training labels
            X_test_path: Path to test features
            y_test_path: Path to test labels
        """
        self.X_train = pd.read_csv(X_train_path)
        self.y_train = pd.read_csv(y_train_path).values.ravel()
        self.X_test = pd.read_csv(X_test_path)
        self.y_test = pd.read_csv(y_test_path).values.ravel()
        
        self.model = None
        self.predictions = None
        self.probabilities = None
        self.metrics = {}
        
        print(f"\n{'='*80}")
        print(f"PHASE 5: XGBOOST MODEL TRAINING")
        print(f"{'='*80}")
        print(f"\n📊 Data Loaded:")
        print(f"   Training: {len(self.X_train)} samples × {self.X_train.shape[1]} features")
        print(f"   Test: {len(self.X_test)} samples × {self.X_test.shape[1]} features")
    
    def train_model(self, verbose: bool = True) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier with investor feature prioritization
        
        Args:
            verbose: Print training progress
            
        Returns:
            Trained XGBClassifier model
        """
        print(f"\n{'─'*80}")
        print(f"STEP 1: INITIALIZE & TRAIN XGBOOST MODEL")
        print(f"{'─'*80}")
        
        # Calculate class weights for imbalance handling
        class_0_count = (self.y_train == 0).sum()
        class_1_count = (self.y_train == 1).sum()
        scale_pos_weight = class_0_count / class_1_count
        
        print(f"\nClass Distribution (Training):")
        print(f"  Success (1): {class_1_count} ({class_1_count/len(self.y_train)*100:.1f}%)")
        print(f"  Failure (0): {class_0_count} ({class_0_count/len(self.y_train)*100:.1f}%)")
        print(f"  Scale POS Weight: {scale_pos_weight:.2f}")
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            
            # Class imbalance handling
            scale_pos_weight=scale_pos_weight,
            
            # Regularization (prevent overfitting on small dataset)
            max_depth=4,
            reg_alpha=1.0,      # L1 regularization
            reg_lambda=1.0,     # L2 regularization
            
            # Learning parameters
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            
            # Other
            random_state=42,
            n_jobs=-1,          # Use all CPU cores
            verbose=0
        )
        
        # Train
        print(f"\nTraining XGBoost model...")
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)]
        )
        
        print(f"✓ Model trained successfully")
        print(f"  Boosting rounds: {self.model.n_estimators}")
        print(f"  Max tree depth: {self.model.max_depth}")
        
        return self.model
    
    def evaluate_model(self) -> dict:
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'─'*80}")
        print(f"STEP 2: MODEL EVALUATION")
        print(f"{'─'*80}")
        
        # Predictions
        self.predictions = self.model.predict(self.X_test)
        self.probabilities = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(self.y_test, self.probabilities)
        f1 = f1_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions, zero_division=0)
        recall = recall_score(self.y_test, self.predictions, zero_division=0)
        accuracy = (self.predictions == self.y_test).mean()
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        self.metrics = {
            'roc_auc': roc_auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'specificity': specificity,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        print(f"\n📊 PERFORMANCE METRICS (Test Set):")
        print(f"{'─'*50}")
        print(f"  ROC-AUC Score:        {roc_auc:.4f} ⭐")
        print(f"  F1-Score:             {f1:.4f}")
        print(f"  Accuracy:             {accuracy:.4f}")
        print(f"  Precision:            {precision:.4f} (quality of positive predictions)")
        print(f"  Recall:               {recall:.4f} (catch true positives)")
        print(f"  Specificity:          {specificity:.4f} (avoid false positives)")
        
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"{'─'*50}")
        print(f"  True Positives:       {tp} (correctly predicted success)")
        print(f"  True Negatives:       {tn} (correctly predicted failure)")
        print(f"  False Positives:      {fp} (wrongly predicted success)")
        print(f"  False Negatives:      {fn} (wrongly predicted failure)")
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from model
        
        Returns:
            DataFrame with features ranked by importance
        """
        print(f"\n{'─'*80}")
        print(f"STEP 3: FEATURE IMPORTANCE (FROM MODEL)")
        print(f"{'─'*80}")
        
        # Get importances
        importances = self.model.feature_importances_
        feature_names = self.X_train.columns
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'category': self._categorize_features(feature_names)
        }).sort_values('importance', ascending=False)
        
        # Investor features analysis
        investor_features = importance_df[importance_df['category'] == 'Investor']
        
        print(f"\n👤 TOP 15 FEATURES (Model-Learned Importance):")
        print(f"{'─'*70}")
        print(f"{'Rank':<5} {'Feature':<40} {'Importance':<15} {'Category':<12}")
        print(f"{'─'*70}")
        
        for idx, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            print(f"{idx:<5} {row['feature']:<40} {row['importance']:<15.4f} {row['category']:<12}")
        
        print(f"\n📊 INVESTOR FEATURES IN MODEL:")
        if len(investor_features) > 0:
            print(f"  Total Importance: {investor_features['importance'].sum():.4f}")
            print(f"  Average Importance: {investor_features['importance'].mean():.4f}")
            print(f"  In Top 10: {len(investor_features[investor_features['importance'] >= importance_df.iloc[9]['importance']])}/6")
            print(f"\n  Ranked:")
            for idx, (_, row) in enumerate(investor_features.iterrows(), 1):
                rank = len(importance_df[importance_df['importance'] > row['importance']]) + 1
                print(f"    #{rank}: {row['feature']:<40} ({row['importance']:.4f})")
        
        return importance_df
    
    def _categorize_features(self, features) -> list:
        """Categorize features by type"""
        investor_features = [
            'investor_risk_capacity', 'investor_risk_tolerance',
            'investor_behavioral_fragility', 'investor_time_horizon_strength',
            'investor_effective_risk_tolerance', 'investor_time_horizon_years'
        ]
        asset_features = [
            'asset_returns_60d_ma', 'asset_volatility_30d', 'asset_sharpe_ratio',
            'asset_sortino_ratio', 'asset_calmar_ratio', 'asset_max_drawdown',
            'asset_skewness', 'asset_kurtosis', 'asset_beta'
        ]
        market_features = [
            'market_vix', 'market_volatility_level', 'market_vix_percentile',
            'nifty50_level', 'market_return_1m', 'market_return_3m',
            'market_regime_bull', 'market_regime_bear', 'risk_free_rate',
            'market_top_sector_return', 'market_bottom_sector_return',
            'market_sector_return_dispersion'
        ]
        
        categories = []
        for f in features:
            if f in investor_features:
                categories.append('Investor')
            elif f in asset_features:
                categories.append('Asset')
            elif f in market_features:
                categories.append('Market')
            else:
                categories.append('Portfolio')
        return categories
    
    def cross_validate(self, cv: int = 5) -> dict:
        """
        Perform k-fold cross-validation
        
        Args:
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        print(f"\n{'─'*80}")
        print(f"STEP 4: CROSS-VALIDATION ({cv}-FOLD)")
        print(f"{'─'*80}")
        
        # Combine train + test for full CV
        X_combined = pd.concat([self.X_train, self.X_test], ignore_index=True)
        y_combined = np.concatenate([self.y_train, self.y_test])
        
        # Create fresh model for CV
        cv_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=(y_combined == 0).sum() / (y_combined == 1).sum(),
            max_depth=4,
            reg_alpha=1.0,
            reg_lambda=1.0,
            learning_rate=0.05,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # CV scores
        cv_scores = cross_val_score(cv_model, X_combined, y_combined, cv=cv, scoring='roc_auc')
        
        print(f"\n  Fold Scores: {cv_scores}")
        print(f"  Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return {
            'cv_scores': cv_scores,
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std()
        }
    
    def save_model(self, model_path: str = None) -> str:
        """
        Save trained model to disk
        
        Args:
            model_path: Path to save model
            
        Returns:
            Path to saved model
        """
        if model_path is None:
            model_path = "f:\\AI Insights Dashboard\\trained_xgboost_model.json"
        
        self.model.save_model(model_path)
        print(f"\n✅ Model saved: {model_path}")
        
        return model_path
    
    def save_metrics(self, metrics_path: str = None) -> str:
        """
        Save metrics to JSON
        
        Args:
            metrics_path: Path to save metrics
            
        Returns:
            Path to saved metrics
        """
        if metrics_path is None:
            metrics_path = "f:\\AI Insights Dashboard\\model_metrics.json"
        
        metrics_to_save = {
            'phase': 5,
            'model_type': 'XGBClassifier',
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features': self.X_train.shape[1],
            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in self.metrics.items()},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"✅ Metrics saved: {metrics_path}")
        return metrics_path


def main():
    """Run complete Phase 5 pipeline"""
    
    # Initialize trainer
    trainer = XGBoostModelTrainer()
    
    # Train model
    trainer.train_model()
    
    # Evaluate
    metrics = trainer.evaluate_model()
    
    # Feature importance
    importance_df = trainer.get_feature_importance()
    
    # Cross-validation
    cv_results = trainer.cross_validate(cv=5)
    
    # Save outputs
    model_path = trainer.save_model()
    metrics_path = trainer.save_metrics()
    importance_df.to_csv("f:\\AI Insights Dashboard\\feature_importance.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ PHASE 5 COMPLETE: XGBOOST MODEL TRAINING")
    print(f"{'='*80}")
    print(f"\n📋 SUMMARY:")
    print(f"  Model Type: XGBClassifier")
    print(f"  Training Samples: {len(trainer.X_train)}")
    print(f"  Test Samples: {len(trainer.X_test)}")
    print(f"  Features: {trainer.X_train.shape[1]}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  CV Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    print(f"\n📁 OUTPUT FILES:")
    print(f"  ✓ {model_path}")
    print(f"  ✓ {metrics_path}")
    print(f"  ✓ feature_importance.csv")
    print(f"\n🎯 INVESTOR FEATURE IMPACT:")
    investor_imp = importance_df[importance_df['category'] == 'Investor']
    if len(investor_imp) > 0:
        print(f"  Total: {investor_imp['importance'].sum():.4f}")
        print(f"  Top: {investor_imp.iloc[0]['feature']} ({investor_imp.iloc[0]['importance']:.4f})")
    print(f"\nReady for Phase 6: Model Deployment")


if __name__ == "__main__":
    main()
