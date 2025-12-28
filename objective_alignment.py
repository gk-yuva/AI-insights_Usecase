"""
Objective Alignment Module
Evaluates portfolio alignment with investment objectives
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from enum import Enum


class InvestmentObjective(Enum):
    """Investment objective types"""
    CONSERVATIVE = "Conservative Income"
    MODERATE = "Moderate Growth"
    AGGRESSIVE = "Aggressive Growth"
    BALANCED = "Balanced"


class ObjectiveAlignmentAnalyzer:
    """Analyze portfolio alignment with investment objectives"""
    
    # Define thresholds for each objective
    OBJECTIVE_CRITERIA = {
        InvestmentObjective.CONSERVATIVE: {
            'min_return': 5.0,      # Minimum 5% annual return
            'max_return': 12.0,     # Target return
            'max_var_95': 15.0,     # Max VaR 95%
            'min_sharpe': 0.5,      # Minimum Sharpe ratio
            'max_volatility': 20.0, # Max annual volatility
            'min_sortino': 0.7,
            'description': 'Focus on capital preservation with steady income'
        },
        InvestmentObjective.MODERATE: {
            'min_return': 10.0,     # Minimum 10% annual return
            'max_return': 18.0,     # Target return
            'max_var_95': 25.0,     # Max VaR 95%
            'min_sharpe': 0.7,      # Minimum Sharpe ratio
            'max_volatility': 30.0, # Max annual volatility
            'min_sortino': 0.9,
            'description': 'Balance between growth and risk management'
        },
        InvestmentObjective.AGGRESSIVE: {
            'min_return': 15.0,     # Minimum 15% annual return
            'max_return': 30.0,     # Target return
            'max_var_95': 40.0,     # Max VaR 95%
            'min_sharpe': 0.8,      # Minimum Sharpe ratio
            'max_volatility': 45.0, # Max annual volatility
            'min_sortino': 1.0,
            'description': 'Maximum growth with higher risk tolerance'
        },
        InvestmentObjective.BALANCED: {
            'min_return': 8.0,      # Minimum 8% annual return
            'max_return': 15.0,     # Target return
            'max_var_95': 20.0,     # Max VaR 95%
            'min_sharpe': 0.6,      # Minimum Sharpe ratio
            'max_volatility': 25.0, # Max annual volatility
            'min_sortino': 0.8,
            'description': 'Diversified approach across asset classes'
        }
    }
    
    def __init__(self):
        """Initialize objective alignment analyzer"""
        pass
    
    def parse_objective(self, objective_str: str) -> InvestmentObjective:
        """
        Parse objective string to enum
        
        Args:
            objective_str: Objective as string
            
        Returns:
            InvestmentObjective enum
        """
        objective_map = {
            'conservative': InvestmentObjective.CONSERVATIVE,
            'moderate': InvestmentObjective.MODERATE,
            'aggressive': InvestmentObjective.AGGRESSIVE,
            'balanced': InvestmentObjective.BALANCED,
        }
        
        # Try to match
        obj_lower = objective_str.lower()
        for key, value in objective_map.items():
            if key in obj_lower:
                return value
        
        # Default to moderate
        return InvestmentObjective.MODERATE
    
    def evaluate_alignment(self, 
                          metrics: Dict, 
                          objective: InvestmentObjective) -> Dict:
        """
        Evaluate how well portfolio aligns with objective
        
        Args:
            metrics: Portfolio metrics dictionary
            objective: Investment objective
            
        Returns:
            Alignment evaluation dictionary
        """
        criteria = self.OBJECTIVE_CRITERIA[objective]
        
        # Score each criterion
        scores = {}
        
        # Return alignment
        if metrics['annual_return']:
            return_score = self._score_return(
                metrics['annual_return'], 
                criteria['min_return'], 
                criteria['max_return']
            )
            scores['return'] = return_score
        
        # VaR alignment (lower is better)
        if metrics['var_95']:
            var_score = self._score_var(metrics['var_95'], criteria['max_var_95'])
            scores['var'] = var_score
        
        # Sharpe ratio alignment (higher is better)
        if metrics['sharpe_ratio']:
            sharpe_score = self._score_sharpe(metrics['sharpe_ratio'], criteria['min_sharpe'])
            scores['sharpe'] = sharpe_score
        
        # Volatility alignment (lower is better for conservative)
        if metrics['volatility']:
            vol_score = self._score_volatility(metrics['volatility'], criteria['max_volatility'])
            scores['volatility'] = vol_score
        
        # Sortino ratio alignment
        if metrics['sortino_ratio']:
            sortino_score = self._score_sortino(metrics['sortino_ratio'], criteria['min_sortino'])
            scores['sortino'] = sortino_score
        
        # Calculate overall alignment score (0-100)
        overall_score = np.mean(list(scores.values())) if scores else 0
        
        # Determine alignment category
        alignment_category = self._categorize_alignment(overall_score)
        
        return {
            'objective': objective.value,
            'overall_score': overall_score,
            'alignment_category': alignment_category,
            'individual_scores': scores,
            'criteria': criteria,
            'recommendations': self._generate_recommendations(metrics, criteria, scores)
        }
    
    def _score_return(self, actual: float, min_target: float, max_target: float) -> float:
        """Score return performance (0-100)"""
        if actual >= max_target:
            return 100
        elif actual >= min_target:
            # Linear scale between min and max
            return 50 + 50 * ((actual - min_target) / (max_target - min_target))
        else:
            # Below minimum
            return max(0, 50 * (actual / min_target))
    
    def _score_var(self, actual: float, max_acceptable: float) -> float:
        """Score VaR (lower is better, 0-100)"""
        if actual <= max_acceptable:
            return 100 - (actual / max_acceptable) * 30  # Higher score for lower VaR
        else:
            # Penalty for exceeding threshold
            excess = (actual - max_acceptable) / max_acceptable
            return max(0, 70 - excess * 100)
    
    def _score_sharpe(self, actual: float, min_target: float) -> float:
        """Score Sharpe ratio (0-100)"""
        if actual >= min_target * 1.5:  # Excellent
            return 100
        elif actual >= min_target:
            return 70 + 30 * ((actual - min_target) / (min_target * 0.5))
        else:
            return max(0, 70 * (actual / min_target))
    
    def _score_volatility(self, actual: float, max_acceptable: float) -> float:
        """Score volatility (lower is better, 0-100)"""
        if actual <= max_acceptable * 0.7:  # Well below threshold
            return 100
        elif actual <= max_acceptable:
            return 70 + 30 * (1 - (actual - max_acceptable * 0.7) / (max_acceptable * 0.3))
        else:
            # Penalty for high volatility
            excess = (actual - max_acceptable) / max_acceptable
            return max(0, 70 - excess * 100)
    
    def _score_sortino(self, actual: float, min_target: float) -> float:
        """Score Sortino ratio (0-100)"""
        if actual >= min_target * 1.5:
            return 100
        elif actual >= min_target:
            return 70 + 30 * ((actual - min_target) / (min_target * 0.5))
        else:
            return max(0, 70 * (actual / min_target))
    
    def _categorize_alignment(self, score: float) -> str:
        """Categorize alignment based on score"""
        if score >= 80:
            return "Excellent Alignment"
        elif score >= 65:
            return "Good Alignment"
        elif score >= 50:
            return "Moderate Alignment"
        elif score >= 35:
            return "Poor Alignment"
        else:
            return "Misaligned"
    
    def _generate_recommendations(self, 
                                 metrics: Dict, 
                                 criteria: Dict, 
                                 scores: Dict) -> list:
        """Generate specific recommendations based on gaps"""
        recommendations = []
        
        # Return recommendations
        if 'return' in scores and scores['return'] < 70:
            if metrics['annual_return'] < criteria['min_return']:
                recommendations.append(
                    f"Consider higher-growth assets to meet {criteria['min_return']}% return target"
                )
        
        # Risk recommendations
        if 'var' in scores and scores['var'] < 70:
            if metrics['var_95'] > criteria['max_var_95']:
                recommendations.append(
                    f"Portfolio VaR ({metrics['var_95']:.2f}%) exceeds {criteria['max_var_95']}% threshold. "
                    "Consider reducing high-risk positions or adding defensive assets"
                )
        
        # Sharpe ratio recommendations
        if 'sharpe' in scores and scores['sharpe'] < 70:
            recommendations.append(
                "Improve risk-adjusted returns by optimizing position sizing or diversification"
            )
        
        # Volatility recommendations
        if 'volatility' in scores and scores['volatility'] < 70:
            if metrics['volatility'] > criteria['max_volatility']:
                recommendations.append(
                    f"Reduce portfolio volatility ({metrics['volatility']:.2f}%) through diversification or "
                    "lower-volatility assets"
                )
        
        if not recommendations:
            recommendations.append("Portfolio is well-aligned with investment objective")
        
        return recommendations
