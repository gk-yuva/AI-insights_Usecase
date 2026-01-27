"""
Production Monitoring for ML Model

Tracks model performance, data drift, prediction drift, and other
production metrics to ensure model quality over time.

Features:
- Data drift detection (Kolmogorov-Smirnov test)
- Model performance monitoring
- Prediction distribution tracking
- Automatic retraining triggers
- Performance alerting

Phase 6: Model Deployment & Integration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import stats
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Production monitoring for ML model
    
    Tracks:
    - Model performance (ROC-AUC, F1, accuracy) on recent predictions
    - Data drift (input distribution changes)
    - Model drift (prediction distribution changes)
    - Inference latency
    - Prediction confidence distribution
    """
    
    def __init__(
        self,
        baseline_auc: float = 0.9853,
        baseline_f1: float = 0.9375,
        monitor_dir: str = './monitoring_logs',
        alert_threshold: float = 0.95
    ):
        """
        Initialize model monitor
        
        Args:
            baseline_auc: Baseline ROC-AUC from training (0.9853)
            baseline_f1: Baseline F1-score from training (0.9375)
            monitor_dir: Directory for monitoring logs
            alert_threshold: Performance degradation threshold for alerts (0.95 = 5% degradation)
        """
        self.baseline_auc = baseline_auc
        self.baseline_f1 = baseline_f1
        self.alert_threshold = alert_threshold
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking lists
        self.predictions = []
        self.actuals = []
        self.features = []
        self.timestamps = []
        self.latencies = []
        self.confidences = []
        
        # Baseline distributions (from training data)
        self.baseline_feature_means = None
        self.baseline_feature_stds = None
        self.baseline_prediction_mean = None
        self.baseline_prediction_std = None
        
        # Alerts log
        self.alerts = []
        
        logger.info(f"✅ Model Monitor initialized")
        logger.info(f"   Baseline ROC-AUC: {baseline_auc:.4f}")
        logger.info(f"   Baseline F1: {baseline_f1:.4f}")
        logger.info(f"   Alert Threshold: {alert_threshold:.1%}")
    
    def set_baseline_distributions(
        self,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        prediction_mean: float,
        prediction_std: float
    ) -> None:
        """
        Set baseline distributions from training data for drift detection
        
        Args:
            feature_means: Mean of each feature from training
            feature_stds: Std of each feature from training
            prediction_mean: Mean prediction probability from training
            prediction_std: Std prediction probability from training
        """
        self.baseline_feature_means = np.array(feature_means)
        self.baseline_feature_stds = np.array(feature_stds)
        self.baseline_prediction_mean = prediction_mean
        self.baseline_prediction_std = prediction_std
        
        logger.info("✅ Baseline distributions set for drift detection")
    
    def log_prediction(
        self,
        features: np.ndarray,
        prediction_probability: float,
        confidence: float,
        latency_ms: float,
        actual_label: Optional[int] = None,
        asset_symbol: Optional[str] = None
    ) -> None:
        """
        Log a model prediction for monitoring
        
        Args:
            features: [37] feature vector used for prediction
            prediction_probability: Model output probability [0,1]
            confidence: Model confidence score [0,1]
            latency_ms: Inference time in milliseconds
            actual_label: True label if available (for performance tracking)
            asset_symbol: Asset being predicted on
        """
        self.features.append(features)
        self.predictions.append(prediction_probability)
        self.timestamps.append(datetime.now())
        self.latencies.append(latency_ms)
        self.confidences.append(confidence)
        
        if actual_label is not None:
            self.actuals.append(actual_label)
        
        # Log periodically
        if len(self.predictions) % 100 == 0:
            self._save_logs()
            self._check_all_metrics()
    
    def check_data_drift(self, window_size: int = 100) -> Dict[str, any]:
        """
        Check for data drift (input distribution changes)
        
        Uses Kolmogorov-Smirnov test on recent vs baseline
        
        Args:
            window_size: Number of recent predictions to analyze
        
        Returns:
            {
                'drifted': bool,
                'n_drifted_features': int,
                'drifted_features': list of feature names,
                'ks_statistics': dict,
                'p_values': dict
            }
        """
        if len(self.features) < window_size or self.baseline_feature_means is None:
            return {
                'status': 'insufficient_data',
                'n_predictions': len(self.features),
                'window_size': window_size
            }
        
        recent_features = np.array(self.features[-window_size:])
        
        ks_stats = {}
        p_values = {}
        drifted_features = []
        
        for i in range(recent_features.shape[1]):
            # Normalize recent features by baseline
            recent_normalized = (
                (recent_features[:, i] - self.baseline_feature_means[i]) /
                (self.baseline_feature_stds[i] + 1e-6)
            )
            
            # KS test: normalized vs standard normal
            ks_stat, p_val = stats.kstest(recent_normalized, 'norm')
            
            ks_stats[i] = float(ks_stat)
            p_values[i] = float(p_val)
            
            # Detect drift (p < 0.05 = significant difference)
            if p_val < 0.05:
                drifted_features.append(i)
        
        drifted = len(drifted_features) > 0
        
        result = {
            'drifted': drifted,
            'n_drifted_features': len(drifted_features),
            'drifted_feature_indices': drifted_features,
            'ks_statistics': ks_stats,
            'p_values': p_values,
            'window_size': window_size
        }
        
        if drifted:
            self._alert(f"⚠️  DATA DRIFT DETECTED: {len(drifted_features)} features drifted", result)
        
        return result
    
    def check_prediction_drift(self, window_size: int = 100) -> Dict[str, any]:
        """
        Check for model drift (prediction distribution changes)
        
        Args:
            window_size: Number of recent predictions to analyze
        
        Returns:
            {
                'drifted': bool,
                'mean_shift': float,
                'std_shift': float,
                'recent_mean': float,
                'recent_std': float
            }
        """
        if len(self.predictions) < window_size or self.baseline_prediction_mean is None:
            return {'status': 'insufficient_data'}
        
        recent_predictions = np.array(self.predictions[-window_size:])
        recent_mean = np.mean(recent_predictions)
        recent_std = np.std(recent_predictions)
        
        mean_shift = abs(recent_mean - self.baseline_prediction_mean)
        std_shift = abs(recent_std - self.baseline_prediction_std)
        
        # Drift detected if mean shifts > 1 std or std changes > 50%
        drifted = (mean_shift > self.baseline_prediction_std or 
                  std_shift / (self.baseline_prediction_std + 1e-6) > 0.5)
        
        result = {
            'drifted': drifted,
            'mean_shift': float(mean_shift),
            'std_shift': float(std_shift),
            'recent_mean': float(recent_mean),
            'recent_std': float(recent_std),
            'baseline_mean': float(self.baseline_prediction_mean),
            'baseline_std': float(self.baseline_prediction_std),
            'window_size': window_size
        }
        
        if drifted:
            self._alert(f"⚠️  PREDICTION DRIFT: mean_shift={mean_shift:.4f}, std_shift={std_shift:.4f}", result)
        
        return result
    
    def check_performance_degradation(self, window_size: int = 50) -> Dict[str, any]:
        """
        Check for model performance degradation
        
        Args:
            window_size: Number of recent predictions with actuals to analyze
        
        Returns:
            {
                'degraded': bool,
                'recent_auc': float,
                'recent_f1': float,
                'auc_degradation': float,
                'f1_degradation': float
            }
        """
        if len(self.actuals) < window_size:
            return {
                'status': 'insufficient_data',
                'actuals_available': len(self.actuals),
                'window_size': window_size
            }
        
        recent_preds = np.array(self.predictions[-window_size:])
        recent_actuals = np.array(self.actuals[-window_size:])
        
        try:
            from sklearn.metrics import roc_auc_score, f1_score
            
            recent_auc = roc_auc_score(recent_actuals, recent_preds)
            recent_f1 = f1_score(recent_actuals, (recent_preds > 0.5).astype(int))
            
            auc_degradation = (self.baseline_auc - recent_auc) / self.baseline_auc
            f1_degradation = (self.baseline_f1 - recent_f1) / self.baseline_f1
            
            # Alert if degraded beyond threshold
            degraded = (recent_auc < self.baseline_auc * self.alert_threshold or
                       recent_f1 < self.baseline_f1 * self.alert_threshold)
            
            result = {
                'degraded': degraded,
                'recent_auc': float(recent_auc),
                'recent_f1': float(recent_f1),
                'baseline_auc': float(self.baseline_auc),
                'baseline_f1': float(self.baseline_f1),
                'auc_degradation_pct': float(auc_degradation * 100),
                'f1_degradation_pct': float(f1_degradation * 100),
                'window_size': window_size
            }
            
            if degraded:
                self._alert(
                    f"⚠️  PERFORMANCE DEGRADATION: AUC {recent_auc:.4f} (baseline {self.baseline_auc:.4f})",
                    result
                )
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {'error': str(e)}
    
    def check_inference_latency(self, threshold_ms: float = 500.0) -> Dict[str, any]:
        """
        Check inference latency statistics
        
        Args:
            threshold_ms: Alert threshold in milliseconds
        
        Returns:
            {
                'slow_inference': bool,
                'mean_latency_ms': float,
                'max_latency_ms': float,
                'p95_latency_ms': float
            }
        """
        if len(self.latencies) == 0:
            return {'status': 'no_data'}
        
        latencies = np.array(self.latencies)
        mean_lat = np.mean(latencies)
        max_lat = np.max(latencies)
        p95_lat = np.percentile(latencies, 95)
        
        slow = max_lat > threshold_ms
        
        result = {
            'slow_inference': slow,
            'mean_latency_ms': float(mean_lat),
            'max_latency_ms': float(max_lat),
            'p95_latency_ms': float(p95_lat),
            'threshold_ms': threshold_ms,
            'n_predictions': len(latencies)
        }
        
        if slow:
            self._alert(
                f"⚠️  SLOW INFERENCE: max={max_lat:.1f}ms (threshold {threshold_ms:.1f}ms)",
                result
            )
        
        return result
    
    def check_confidence_distribution(self) -> Dict[str, any]:
        """
        Analyze model confidence score distribution
        
        Returns:
            {
                'mean_confidence': float,
                'std_confidence': float,
                'min_confidence': float,
                'max_confidence': float,
                'low_confidence_pct': float
            }
        """
        if len(self.confidences) == 0:
            return {'status': 'no_data'}
        
        confidences = np.array(self.confidences)
        low_conf_pct = (confidences < 0.5).sum() / len(confidences)
        
        result = {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'low_confidence_pct': float(low_conf_pct),
            'n_predictions': len(confidences)
        }
        
        if low_conf_pct > 0.2:
            self._alert(f"⚠️  LOW CONFIDENCE: {low_conf_pct:.1%} predictions below 0.5", result)
        
        return result
    
    def _check_all_metrics(self) -> None:
        """Run all monitoring checks"""
        logger.info("Running comprehensive monitoring checks...")
        
        data_drift = self.check_data_drift()
        pred_drift = self.check_prediction_drift()
        perf_degrad = self.check_performance_degradation()
        latency = self.check_inference_latency()
        confidence = self.check_confidence_distribution()
        
        # Summarize
        issues = 0
        if data_drift.get('drifted'): issues += 1
        if pred_drift.get('drifted'): issues += 1
        if perf_degrad.get('degraded'): issues += 1
        if latency.get('slow_inference'): issues += 1
        
        if issues > 0:
            logger.warning(f"⚠️  Found {issues} monitoring issues")
        else:
            logger.info("✅ All monitoring checks passed")
    
    def should_trigger_retraining(self) -> Tuple[bool, str]:
        """
        Determine if model should be retrained
        
        Returns:
            (should_retrain, reason)
        """
        reasons = []
        
        # Check performance degradation
        perf = self.check_performance_degradation()
        if perf.get('degraded'):
            reasons.append(f"Performance degraded: AUC {perf.get('recent_auc', 0):.4f}")
        
        # Check data drift
        drift = self.check_data_drift()
        if drift.get('drifted'):
            reasons.append(f"Data drift in {drift.get('n_drifted_features', 0)} features")
        
        # Enough new data?
        if len(self.actuals) > 200:  # 200 new labeled examples
            reasons.append("Sufficient new labeled data accumulated (200+ examples)")
        
        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retraining needed"
        
        return should_retrain, reason
    
    def _alert(self, message: str, details: Dict) -> None:
        """Log an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'details': details
        }
        self.alerts.append(alert)
        logger.warning(message)
    
    def _save_logs(self) -> None:
        """Save monitoring logs to disk"""
        log_file = self.monitor_dir / f"monitoring_{datetime.now().strftime('%Y%m%d')}.log"
        
        try:
            with open(log_file, 'a') as f:
                f.write(f"\n{datetime.now().isoformat()}\n")
                f.write(f"Predictions: {len(self.predictions)}\n")
                f.write(f"Mean latency: {np.mean(self.latencies):.2f}ms\n")
        except Exception as e:
            logger.error(f"Failed to save logs: {e}")
    
    def generate_report(self) -> str:
        """Generate monitoring report"""
        report = []
        report.append("\n" + "="*70)
        report.append("MODEL MONITORING REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Predictions: {len(self.predictions)}")
        report.append(f"Total Actuals: {len(self.actuals)}")
        
        # Performance
        perf = self.check_performance_degradation()
        if 'recent_auc' in perf:
            report.append(f"\nPerformance (last 50 with actuals):")
            report.append(f"  ROC-AUC: {perf['recent_auc']:.4f} (baseline: {perf['baseline_auc']:.4f})")
            report.append(f"  F1-Score: {perf['recent_f1']:.4f} (baseline: {perf['baseline_f1']:.4f})")
        
        # Drift
        drift = self.check_data_drift()
        if 'drifted' in drift:
            report.append(f"\nData Drift: {'YES' if drift['drifted'] else 'NO'}")
            if drift['drifted']:
                report.append(f"  Features: {drift['n_drifted_features']} drifted")
        
        # Alerts
        if self.alerts:
            report.append(f"\nAlerts ({len(self.alerts)}):")
            for alert in self.alerts[-5:]:
                report.append(f"  - {alert['message']}")
        
        report.append("\n" + "="*70)
        return "\n".join(report)
    
    def print_report(self) -> None:
        """Print monitoring report"""
        print(self.generate_report())


# Example usage
if __name__ == "__main__":
    monitor = ModelMonitor(baseline_auc=0.9853, baseline_f1=0.9375)
    
    # Set baseline distributions
    monitor.set_baseline_distributions(
        feature_means=np.random.normal(0.5, 0.1, 37),
        feature_stds=np.ones(37) * 0.2,
        prediction_mean=0.65,
        prediction_std=0.15
    )
    
    # Simulate predictions
    print("\nSimulating predictions and tracking...")
    for i in range(150):
        features = np.random.normal(0.5, 0.2, 37)
        pred = np.random.beta(6, 3)  # Skewed toward higher probability
        
        monitor.log_prediction(
            features=features,
            prediction_probability=pred,
            confidence=pred,
            latency_ms=np.random.normal(50, 10),
            actual_label=1 if pred > 0.5 else 0
        )
    
    # Generate report
    monitor.print_report()
    
    # Check retraining
    should_retrain, reason = monitor.should_trigger_retraining()
    print(f"\nShould retrain: {should_retrain}")
    print(f"Reason: {reason}")
