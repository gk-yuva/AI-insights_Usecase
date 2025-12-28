"""
Portfolio Quality Scoring
Normalizes portfolio metrics into a 0-100 quality score
"""

from typing import Dict
import numpy as np


class PortfolioQualityScorer:
    """Calculate Portfolio Quality Score (PQS) from metrics"""
    
    # Metric thresholds for normalization
    SHARPE_THRESHOLDS = {
        'bad': 0.5,
        'average': 0.8,
        'excellent': 1.2
    }
    
    SORTINO_THRESHOLDS = {
        'bad': 0.7,
        'average': 1.0,
        'excellent': 1.5
    }
    
    JENSEN_THRESHOLDS = {
        'bad': -0.02,      # -2%
        'average': 0.015,  # 1.5%
        'excellent': 0.03  # 3%
    }
    
    RETURN_THRESHOLDS = {
        'bad': 0.05,       # 5%
        'average': 0.12,   # 12%
        'excellent': 0.18  # 18%
    }
    
    # Component weights (must sum to 100)
    WEIGHTS = {
        'sharpe': 30,
        'sortino': 25,
        'jensen_alpha': 20,
        'returns': 15,
        'volatility': 10  # Lower is better
    }
    
    def __init__(self):
        """Initialize portfolio quality scorer"""
        self.component_scores = {}
    
    def _normalize_metric(self, value: float, thresholds: Dict, inverse: bool = False) -> float:
        """
        Normalize a metric to 0-100 scale
        
        Args:
            value: Metric value
            thresholds: Dict with 'bad', 'average', 'excellent' keys
            inverse: True if lower is better (e.g., volatility)
            
        Returns:
            Score 0-100
        """
        if value is None:
            return 50  # Neutral score if missing
        
        bad = thresholds['bad']
        avg = thresholds['average']
        exc = thresholds['excellent']
        
        if inverse:
            # For metrics where lower is better
            if value <= exc:
                return 100
            elif value <= avg:
                # Linear interpolation between excellent and average
                return 100 - 50 * (value - exc) / (avg - exc)
            elif value <= bad:
                # Linear interpolation between average and bad
                return 50 - 50 * (value - avg) / (bad - avg)
            else:
                return max(0, 20 - (value - bad) * 10)  # Penalty for very bad
        else:
            # For metrics where higher is better
            if value >= exc:
                return 100
            elif value >= avg:
                # Linear interpolation between average and excellent
                return 50 + 50 * (value - avg) / (exc - avg)
            elif value >= bad:
                # Linear interpolation between bad and average
                return 50 * (value - bad) / (avg - bad)
            else:
                return max(0, 50 * value / bad)
    
    def calculate_sharpe_score(self, sharpe_ratio: float) -> float:
        """Calculate Sharpe ratio component score (0-100)"""
        score = self._normalize_metric(sharpe_ratio, self.SHARPE_THRESHOLDS)
        self.component_scores['sharpe'] = score
        return score
    
    def calculate_sortino_score(self, sortino_ratio: float) -> float:
        """Calculate Sortino ratio component score (0-100)"""
        score = self._normalize_metric(sortino_ratio, self.SORTINO_THRESHOLDS)
        self.component_scores['sortino'] = score
        return score
    
    def calculate_jensen_score(self, jensen_alpha: float) -> float:
        """Calculate Jensen's alpha component score (0-100)"""
        score = self._normalize_metric(jensen_alpha, self.JENSEN_THRESHOLDS)
        self.component_scores['jensen_alpha'] = score
        return score
    
    def calculate_return_score(self, annual_return: float) -> float:
        """Calculate returns component score (0-100)"""
        score = self._normalize_metric(annual_return, self.RETURN_THRESHOLDS)
        self.component_scores['returns'] = score
        return score
    
    def calculate_volatility_score(self, volatility: float) -> float:
        """
        Calculate volatility component score (0-100)
        Lower volatility = higher score
        """
        vol_thresholds = {
            'bad': 0.25,      # 25% volatility = bad
            'average': 0.15,  # 15% = average
            'excellent': 0.08 # 8% = excellent
        }
        score = self._normalize_metric(volatility, vol_thresholds, inverse=True)
        self.component_scores['volatility'] = score
        return score
    
    def calculate_pqs(self, metrics: Dict) -> Dict:
        """
        Calculate Portfolio Quality Score from all metrics
        
        Args:
            metrics: Dict with keys:
                - annual_return
                - sharpe_ratio
                - sortino_ratio
                - jensen_alpha
                - volatility
                
        Returns:
            Dict with PQS and component breakdown
        """
        # Calculate individual component scores
        sharpe_score = self.calculate_sharpe_score(metrics.get('sharpe_ratio', 0))
        sortino_score = self.calculate_sortino_score(metrics.get('sortino_ratio', 0))
        jensen_score = self.calculate_jensen_score(metrics.get('jensen_alpha', 0))
        return_score = self.calculate_return_score(metrics.get('annual_return', 0))
        vol_score = self.calculate_volatility_score(metrics.get('volatility', 0.15))
        
        # Calculate weighted PQS
        pqs = (
            sharpe_score * self.WEIGHTS['sharpe'] / 100 +
            sortino_score * self.WEIGHTS['sortino'] / 100 +
            jensen_score * self.WEIGHTS['jensen_alpha'] / 100 +
            return_score * self.WEIGHTS['returns'] / 100 +
            vol_score * self.WEIGHTS['volatility'] / 100
        )
        
        # Determine quality category
        if pqs >= 80:
            category = "Excellent"
        elif pqs >= 65:
            category = "Good"
        elif pqs >= 50:
            category = "Average"
        elif pqs >= 35:
            category = "Below Average"
        else:
            category = "Poor"
        
        return {
            'pqs_score': round(pqs, 1),
            'category': category,
            'components': {
                'sharpe': round(sharpe_score, 1),
                'sortino': round(sortino_score, 1),
                'jensen_alpha': round(jensen_score, 1),
                'returns': round(return_score, 1),
                'volatility': round(vol_score, 1)
            },
            'weights': self.WEIGHTS,
            'interpretation': self._get_interpretation(pqs)
        }
    
    def _get_interpretation(self, pqs: float) -> str:
        """Get human-readable interpretation of PQS"""
        if pqs >= 80:
            return "This portfolio demonstrates excellent risk-adjusted returns with strong efficiency."
        elif pqs >= 65:
            return "This portfolio shows good performance with reasonable risk management."
        elif pqs >= 50:
            return "This portfolio has average performance characteristics."
        elif pqs >= 35:
            return "This portfolio shows below-average efficiency and risk-adjusted returns."
        else:
            return "This portfolio has significant performance and risk management issues."
