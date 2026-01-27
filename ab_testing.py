"""
A/B Testing Framework for ML vs Rule-Based Recommendations

Enables gradual rollout of ML model by comparing performance against
rule-based baseline. Supports phased deployment strategy:
- Week 1-2: 20% ML, 80% rule-based
- Week 3-4: 50% ML, 50% rule-based
- Week 5-6: 80% ML, 20% rule-based
- Week 7+: 100% ML (rule-based fallback only)

Phase 6: Model Deployment & Integration
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationMethod(Enum):
    """Recommendation source method"""
    ML = 'ML'
    RULE_BASED = 'RULE_BASED'
    FALLBACK = 'FALLBACK'


class ABTestingFramework:
    """
    A/B testing framework for comparing ML and rule-based recommendations
    
    Features:
    - Configurable ML/rule-based split ratio
    - Outcome tracking and statistical analysis
    - Performance comparison metrics
    - Gradual rollout scheduling
    """
    
    def __init__(
        self,
        ml_ratio: float = 0.2,
        log_dir: str = './ab_test_logs',
        test_id: str = 'phase6_ml_deployment'
    ):
        """
        Initialize A/B testing framework
        
        Args:
            ml_ratio: Fraction of recommendations to use ML (0.0-1.0)
            log_dir: Directory for test logs and results
            test_id: Test identifier for tracking
        """
        self.ml_ratio = ml_ratio
        self.log_dir = Path(log_dir)
        self.test_id = test_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.recommendations = []
        self.outcomes = []
        
        # Load existing data if available
        self._load_existing_data()
        
        logger.info(f"✅ A/B Testing Framework initialized")
        logger.info(f"   ML Ratio: {ml_ratio:.1%}")
        logger.info(f"   Log Directory: {self.log_dir}")
        logger.info(f"   Test ID: {test_id}")
    
    def _load_existing_data(self):
        """Load existing recommendations and outcomes from disk"""
        rec_file = self.log_dir / f"{self.test_id}_recommendations.csv"
        if rec_file.exists():
            try:
                self.recommendations = pd.read_csv(rec_file).to_dict('records')
                logger.info(f"Loaded {len(self.recommendations)} existing recommendations")
            except Exception as e:
                logger.warning(f"Failed to load recommendations: {e}")
    
    def get_method(self) -> RecommendationMethod:
        """
        Randomly select recommendation method based on ML ratio
        
        Returns:
            RecommendationMethod (ML or RULE_BASED)
        """
        if np.random.random() < self.ml_ratio:
            return RecommendationMethod.ML
        else:
            return RecommendationMethod.RULE_BASED
    
    def log_recommendation(
        self,
        asset_symbol: str,
        method: RecommendationMethod,
        score: float,
        ml_score: Optional[float] = None,
        rule_score: Optional[float] = None,
        recommendation: str = 'HOLD',
        investor_id: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log a recommendation for later analysis
        
        Args:
            asset_symbol: Asset being recommended
            method: ML or RULE_BASED
            score: Final recommendation score
            ml_score: ML model score (if applicable)
            rule_score: Rule-based score (if applicable)
            recommendation: BUY/SELL/HOLD
            investor_id: Investor identifier
            portfolio_id: Portfolio identifier
            confidence: Confidence score
            metadata: Additional metadata
        
        Returns:
            Recommendation ID
        """
        rec_id = f"{self.test_id}_{datetime.now().isoformat()}_{asset_symbol}"
        
        recommendation_record = {
            'rec_id': rec_id,
            'timestamp': datetime.now().isoformat(),
            'asset_symbol': asset_symbol,
            'method': method.value,
            'score': float(score),
            'ml_score': float(ml_score) if ml_score is not None else None,
            'rule_score': float(rule_score) if rule_score is not None else None,
            'recommendation': recommendation,
            'investor_id': investor_id,
            'portfolio_id': portfolio_id,
            'confidence': float(confidence) if confidence is not None else None,
            'metadata': metadata or {}
        }
        
        self.recommendations.append(recommendation_record)
        
        # Save to CSV periodically
        if len(self.recommendations) % 100 == 0:
            self._save_recommendations()
        
        logger.debug(f"Logged recommendation: {rec_id} ({method.value})")
        return rec_id
    
    def log_outcome(
        self,
        rec_id: str,
        succeeded: bool,
        outcome_metric: float,
        follow_up_value: Optional[float] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Log outcome of a recommendation (success/failure and magnitude)
        
        Args:
            rec_id: Recommendation ID from log_recommendation()
            succeeded: True if recommendation was successful
            outcome_metric: Continuous outcome metric (e.g., Sharpe improvement)
            follow_up_value: Portfolio value after follow-up period
            notes: Additional notes
        """
        outcome_record = {
            'rec_id': rec_id,
            'timestamp': datetime.now().isoformat(),
            'succeeded': bool(succeeded),
            'outcome_metric': float(outcome_metric),
            'follow_up_value': float(follow_up_value) if follow_up_value is not None else None,
            'notes': notes
        }
        
        self.outcomes.append(outcome_record)
        
        # Save outcomes periodically
        if len(self.outcomes) % 50 == 0:
            self._save_outcomes()
        
        logger.debug(f"Logged outcome for: {rec_id} (success={succeeded})")
    
    def _save_recommendations(self):
        """Save recommendations to CSV"""
        rec_file = self.log_dir / f"{self.test_id}_recommendations.csv"
        try:
            df = pd.DataFrame(self.recommendations)
            df.to_csv(rec_file, index=False)
            logger.info(f"Saved {len(self.recommendations)} recommendations")
        except Exception as e:
            logger.error(f"Failed to save recommendations: {e}")
    
    def _save_outcomes(self):
        """Save outcomes to CSV"""
        outcome_file = self.log_dir / f"{self.test_id}_outcomes.csv"
        try:
            df = pd.DataFrame(self.outcomes)
            df.to_csv(outcome_file, index=False)
            logger.info(f"Saved {len(self.outcomes)} outcomes")
        except Exception as e:
            logger.error(f"Failed to save outcomes: {e}")
    
    def analyze_performance(self, min_outcomes: int = 20) -> Dict[str, any]:
        """
        Analyze and compare ML vs rule-based performance
        
        Args:
            min_outcomes: Minimum outcomes required for analysis
        
        Returns:
            Performance analysis dictionary
        """
        if len(self.outcomes) < min_outcomes:
            logger.warning(
                f"Not enough outcomes for analysis: {len(self.outcomes)} < {min_outcomes}"
            )
            return {
                'status': 'insufficient_data',
                'outcomes_logged': len(self.outcomes),
                'recommendations_logged': len(self.recommendations)
            }
        
        # Merge recommendations with outcomes
        outcome_df = pd.DataFrame(self.outcomes)
        rec_df = pd.DataFrame(self.recommendations)
        
        merged = rec_df.merge(outcome_df, on='rec_id', how='inner')
        
        if len(merged) == 0:
            return {
                'status': 'no_matched_data',
                'recommendations': len(rec_df),
                'outcomes': len(outcome_df)
            }
        
        # Split by method
        ml_data = merged[merged['method'] == 'ML']
        rule_data = merged[merged['method'] == 'RULE_BASED']
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_outcomes': len(merged),
            'ml_outcomes': len(ml_data),
            'rule_outcomes': len(rule_data)
        }
        
        # ML Performance
        if len(ml_data) > 0:
            ml_success = ml_data['succeeded'].sum() / len(ml_data)
            ml_avg_metric = ml_data['outcome_metric'].mean()
            ml_std_metric = ml_data['outcome_metric'].std()
            
            results['ml_performance'] = {
                'success_rate': float(ml_success),
                'mean_outcome_metric': float(ml_avg_metric),
                'std_outcome_metric': float(ml_std_metric),
                'n': len(ml_data)
            }
        
        # Rule-Based Performance
        if len(rule_data) > 0:
            rule_success = rule_data['succeeded'].sum() / len(rule_data)
            rule_avg_metric = rule_data['outcome_metric'].mean()
            rule_std_metric = rule_data['outcome_metric'].std()
            
            results['rule_performance'] = {
                'success_rate': float(rule_success),
                'mean_outcome_metric': float(rule_avg_metric),
                'std_outcome_metric': float(rule_std_metric),
                'n': len(rule_data)
            }
        
        # Comparison
        if len(ml_data) > 0 and len(rule_data) > 0:
            ml_succ = ml_data['succeeded'].sum() / len(ml_data)
            rule_succ = rule_data['succeeded'].sum() / len(rule_data)
            improvement = (ml_succ - rule_succ) / rule_succ if rule_succ > 0 else 0
            
            results['comparison'] = {
                'ml_better': bool(ml_succ > rule_succ),
                'success_rate_difference': float(ml_succ - rule_succ),
                'relative_improvement': float(improvement),
                'recommendation': self._get_recommendation(improvement)
            }
        
        return results
    
    def _get_recommendation(self, relative_improvement: float) -> str:
        """Get deployment recommendation based on performance improvement"""
        if relative_improvement > 0.10:
            return "INCREASE_ML_RATIO_AGGRESSIVELY"
        elif relative_improvement > 0.05:
            return "INCREASE_ML_RATIO"
        elif relative_improvement > -0.05:
            return "MAINTAIN_CURRENT_RATIO"
        elif relative_improvement > -0.10:
            return "DECREASE_ML_RATIO"
        else:
            return "SWITCH_BACK_TO_RULE_BASED"
    
    def update_ml_ratio(self, new_ratio: float) -> None:
        """
        Update ML deployment ratio
        
        Args:
            new_ratio: New fraction of ML recommendations (0.0-1.0)
        """
        old_ratio = self.ml_ratio
        self.ml_ratio = np.clip(new_ratio, 0.0, 1.0)
        
        logger.info(f"Updated ML ratio: {old_ratio:.1%} → {self.ml_ratio:.1%}")
    
    def get_gradual_rollout_schedule(self) -> List[Tuple[int, float, str]]:
        """
        Get recommended gradual rollout schedule
        
        Returns:
            List of (week, ml_ratio, status) tuples
        """
        return [
            (1, 0.20, 'MONITOR'),
            (3, 0.50, 'MONITOR'),
            (5, 0.80, 'MONITOR'),
            (7, 1.00, 'FULL_DEPLOYMENT'),
        ]
    
    def print_performance_report(self) -> None:
        """Print formatted performance analysis report"""
        analysis = self.analyze_performance()
        
        print("\n" + "="*70)
        print("A/B TEST PERFORMANCE REPORT")
        print("="*70)
        print(f"Test ID: {self.test_id}")
        print(f"Timestamp: {analysis.get('timestamp', 'N/A')}")
        print(f"Total Outcomes: {analysis.get('total_outcomes', 0)}")
        
        if 'ml_performance' in analysis:
            ml_perf = analysis['ml_performance']
            print(f"\n📊 ML Performance:")
            print(f"   Success Rate: {ml_perf['success_rate']:.1%}")
            print(f"   Mean Metric: {ml_perf['mean_outcome_metric']:.6f}")
            print(f"   Std Metric: {ml_perf['std_outcome_metric']:.6f}")
            print(f"   N Outcomes: {ml_perf['n']}")
        
        if 'rule_performance' in analysis:
            rule_perf = analysis['rule_performance']
            print(f"\n📊 Rule-Based Performance:")
            print(f"   Success Rate: {rule_perf['success_rate']:.1%}")
            print(f"   Mean Metric: {rule_perf['mean_outcome_metric']:.6f}")
            print(f"   Std Metric: {rule_perf['std_outcome_metric']:.6f}")
            print(f"   N Outcomes: {rule_perf['n']}")
        
        if 'comparison' in analysis:
            comp = analysis['comparison']
            print(f"\n🎯 Comparison:")
            print(f"   ML Better: {comp['ml_better']}")
            print(f"   Difference: {comp['success_rate_difference']:+.1%}")
            print(f"   Improvement: {comp['relative_improvement']:+.1%}")
            print(f"   ⚠️  {comp['recommendation']}")
        
        print("\n" + "="*70)
    
    def save_analysis(self, analysis: Dict) -> str:
        """
        Save analysis results to JSON
        
        Args:
            analysis: Analysis dictionary from analyze_performance()
        
        Returns:
            Path to saved analysis file
        """
        analysis_file = self.log_dir / f"{self.test_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            logger.info(f"Saved analysis: {analysis_file}")
            return str(analysis_file)
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            return ""


# Example usage
if __name__ == "__main__":
    # Initialize A/B testing framework
    ab_test = ABTestingFramework(ml_ratio=0.3, test_id='phase6_example')
    
    # Simulate recommendations and outcomes
    print("\n" + "="*60)
    print("SIMULATING RECOMMENDATIONS AND OUTCOMES")
    print("="*60)
    
    for i in range(50):
        # Get recommendation method
        method = ab_test.get_method()
        
        # Simulate score
        if method == RecommendationMethod.ML:
            score = np.random.normal(0.7, 0.2)
        else:
            score = np.random.normal(0.6, 0.25)
        
        # Log recommendation
        rec_id = ab_test.log_recommendation(
            asset_symbol=f"ASSET_{i}",
            method=method,
            score=score,
            recommendation='BUY' if score > 0.5 else 'SELL',
            investor_id=f"INV_{i % 5}",
            confidence=abs(np.random.normal(0.7, 0.15))
        )
        
        # Simulate outcome (30-50% follow-through, ML slightly better)
        if method == RecommendationMethod.ML:
            succeeded = np.random.random() < 0.55
        else:
            succeeded = np.random.random() < 0.50
        
        outcome_metric = np.random.normal(0.1, 0.05) if succeeded else np.random.normal(-0.05, 0.08)
        
        # Log outcome
        ab_test.log_outcome(
            rec_id=rec_id,
            succeeded=succeeded,
            outcome_metric=outcome_metric
        )
        
        if (i + 1) % 10 == 0:
            print(f"  ✓ Logged {i + 1} recommendation-outcome pairs")
    
    # Analyze performance
    print("\n" + "="*60)
    print("ANALYZING PERFORMANCE")
    print("="*60)
    analysis = ab_test.analyze_performance()
    ab_test.print_performance_report()
    ab_test.save_analysis(analysis)
    
    # Print rollout schedule
    print("\n" + "="*60)
    print("GRADUAL ROLLOUT SCHEDULE")
    print("="*60)
    for week, ratio, status in ab_test.get_gradual_rollout_schedule():
        print(f"Week {week}: {ratio:.0%} ML ({status})")
