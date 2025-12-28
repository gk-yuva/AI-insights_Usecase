"""
Portfolio Health Classifier
Combines all metrics to provide overall portfolio health assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from enum import Enum


class HealthStatus(Enum):
    """Portfolio health status categories"""
    HEALTHY = "Healthy"
    WARNING = "Warning"
    AT_RISK = "At Risk"
    CRITICAL = "Critical"


class PortfolioHealthClassifier:
    """Classify overall portfolio health based on multiple factors"""
    
    def __init__(self):
        """Initialize health classifier"""
        pass
    
    def classify_health(self, 
                       metrics: Dict,
                       benchmark_comparison: Dict,
                       objective_alignment: Dict) -> Dict:
        """
        Classify overall portfolio health
        
        Args:
            metrics: Portfolio metrics
            benchmark_comparison: Benchmark comparison results
            objective_alignment: Objective alignment analysis
            
        Returns:
            Health classification dictionary
        """
        # Calculate component health scores
        performance_score = self._assess_performance(metrics, benchmark_comparison)
        risk_score = self._assess_risk(metrics)
        alignment_score = objective_alignment['overall_score']
        
        # Weight the scores
        # Performance: 35%, Risk: 35%, Alignment: 30%
        overall_health_score = (
            performance_score * 0.35 +
            risk_score * 0.35 +
            alignment_score * 0.30
        )
        
        # Determine health status
        health_status = self._determine_status(overall_health_score)
        
        # Identify key issues
        issues = self._identify_issues(metrics, benchmark_comparison, objective_alignment)
        
        # Generate action items
        action_items = self._generate_action_items(
            health_status, issues, metrics, objective_alignment
        )
        
        return {
            'overall_health_score': overall_health_score,
            'health_status': health_status.value,
            'component_scores': {
                'performance': performance_score,
                'risk_management': risk_score,
                'objective_alignment': alignment_score
            },
            'key_issues': issues,
            'action_items': action_items,
            'summary': self._generate_summary(health_status, overall_health_score, issues)
        }
    
    def _assess_performance(self, metrics: Dict, benchmark_comp: Dict) -> float:
        """
        Assess performance component (0-100)
        
        Factors:
        - Absolute returns
        - Benchmark outperformance
        - Sharpe ratio
        - Jensen's alpha
        """
        scores = []
        
        # Absolute return score
        if metrics['annual_return']:
            ret = metrics['annual_return']
            if ret >= 20:
                scores.append(100)
            elif ret >= 15:
                scores.append(90)
            elif ret >= 10:
                scores.append(75)
            elif ret >= 5:
                scores.append(60)
            elif ret >= 0:
                scores.append(40)
            else:
                scores.append(20)
        
        # Benchmark comparison
        if benchmark_comp and benchmark_comp['outperformance'] is not None:
            outperf = benchmark_comp['outperformance']
            if outperf >= 5:
                scores.append(100)
            elif outperf >= 2:
                scores.append(85)
            elif outperf >= 0:
                scores.append(70)
            elif outperf >= -2:
                scores.append(50)
            else:
                scores.append(30)
        
        # Sharpe ratio
        if metrics['sharpe_ratio']:
            sharpe = metrics['sharpe_ratio']
            if sharpe >= 2.0:
                scores.append(100)
            elif sharpe >= 1.5:
                scores.append(90)
            elif sharpe >= 1.0:
                scores.append(75)
            elif sharpe >= 0.5:
                scores.append(60)
            else:
                scores.append(40)
        
        return np.mean(scores) if scores else 50
    
    def _assess_risk(self, metrics: Dict) -> float:
        """
        Assess risk management component (0-100)
        Higher score = better risk management
        
        Factors:
        - VaR levels
        - Maximum drawdown
        - Volatility
        - Sortino ratio
        """
        scores = []
        
        # VaR assessment (lower is better)
        if metrics['var_95']:
            var = metrics['var_95']
            if var <= 15:
                scores.append(100)
            elif var <= 25:
                scores.append(80)
            elif var <= 35:
                scores.append(60)
            elif var <= 45:
                scores.append(40)
            else:
                scores.append(20)
        
        # Maximum drawdown (lower is better)
        if metrics['max_drawdown']:
            mdd = metrics['max_drawdown']
            if mdd <= 10:
                scores.append(100)
            elif mdd <= 20:
                scores.append(80)
            elif mdd <= 30:
                scores.append(60)
            elif mdd <= 40:
                scores.append(40)
            else:
                scores.append(20)
        
        # Volatility (relative assessment)
        if metrics['volatility']:
            vol = metrics['volatility']
            if vol <= 15:
                scores.append(100)
            elif vol <= 25:
                scores.append(85)
            elif vol <= 35:
                scores.append(70)
            elif vol <= 45:
                scores.append(50)
            else:
                scores.append(30)
        
        # Sortino ratio (higher is better)
        if metrics['sortino_ratio']:
            sortino = metrics['sortino_ratio']
            if sortino >= 2.0:
                scores.append(100)
            elif sortino >= 1.5:
                scores.append(85)
            elif sortino >= 1.0:
                scores.append(70)
            elif sortino >= 0.5:
                scores.append(55)
            else:
                scores.append(40)
        
        return np.mean(scores) if scores else 50
    
    def _determine_status(self, overall_score: float) -> HealthStatus:
        """Determine health status from score"""
        if overall_score >= 75:
            return HealthStatus.HEALTHY
        elif overall_score >= 55:
            return HealthStatus.WARNING
        elif overall_score >= 35:
            return HealthStatus.AT_RISK
        else:
            return HealthStatus.CRITICAL
    
    def _identify_issues(self, 
                        metrics: Dict,
                        benchmark_comp: Dict,
                        objective_alignment: Dict) -> List[str]:
        """Identify specific issues with the portfolio"""
        issues = []
        
        # Performance issues
        if metrics['annual_return'] and metrics['annual_return'] < 5:
            issues.append("Low absolute returns (below 5% annually)")
        
        if benchmark_comp and benchmark_comp['outperformance']:
            if benchmark_comp['outperformance'] < -5:
                issues.append(f"Significantly underperforming benchmark by {abs(benchmark_comp['outperformance']):.2f}%")
        
        # Risk issues
        if metrics['var_95'] and metrics['var_95'] > 35:
            issues.append(f"High Value at Risk: {metrics['var_95']:.2f}% (95% confidence)")
        
        if metrics['max_drawdown'] and metrics['max_drawdown'] > 30:
            issues.append(f"Large maximum drawdown: {metrics['max_drawdown']:.2f}%")
        
        if metrics['sharpe_ratio'] and metrics['sharpe_ratio'] < 0.5:
            issues.append("Poor risk-adjusted returns (Sharpe ratio below 0.5)")
        
        # Alignment issues
        if objective_alignment['overall_score'] < 50:
            issues.append(f"Portfolio misaligned with {objective_alignment['objective']} objective")
        
        # Negative alpha
        if metrics.get('jensens_alpha') and metrics['jensens_alpha'] < -2:
            issues.append(f"Negative Jensen's Alpha: {metrics['jensens_alpha']:.2f}%")
        
        return issues
    
    def _generate_action_items(self,
                               health_status: HealthStatus,
                               issues: List[str],
                               metrics: Dict,
                               objective_alignment: Dict) -> List[str]:
        """Generate specific action items based on health status"""
        actions = []
        
        if health_status == HealthStatus.CRITICAL:
            actions.append("🚨 URGENT: Review and restructure portfolio immediately")
            actions.append("Consider reducing position sizes in high-risk assets")
            actions.append("Consult with financial advisor for portfolio recovery strategy")
        
        elif health_status == HealthStatus.AT_RISK:
            actions.append("⚠️ Review portfolio allocation within next week")
            actions.append("Identify and reduce positions with excessive volatility")
        
        elif health_status == HealthStatus.WARNING:
            actions.append("📊 Monitor portfolio closely and consider rebalancing")
        
        # Specific actions based on issues
        if metrics['var_95'] and metrics['var_95'] > 30:
            actions.append("Reduce overall portfolio risk through diversification or hedging")
        
        if 'outperformance' in str(issues).lower() and 'under' in str(issues).lower():
            actions.append("Review underperforming holdings and consider replacements")
        
        # Add objective alignment recommendations
        if objective_alignment.get('recommendations'):
            actions.extend(objective_alignment['recommendations'][:2])  # Top 2 recommendations
        
        if not actions:
            actions.append("✅ Continue current strategy with regular monitoring")
            actions.append("Consider rebalancing if allocation drifts more than 5%")
        
        return actions
    
    def _generate_summary(self, 
                         health_status: HealthStatus,
                         overall_score: float,
                         issues: List[str]) -> str:
        """Generate human-readable summary"""
        status_descriptions = {
            HealthStatus.HEALTHY: "Portfolio is performing well with appropriate risk levels",
            HealthStatus.WARNING: "Portfolio shows some areas of concern requiring attention",
            HealthStatus.AT_RISK: "Portfolio has significant issues that need addressing",
            HealthStatus.CRITICAL: "Portfolio requires immediate intervention"
        }
        
        summary = f"{status_descriptions[health_status]} (Health Score: {overall_score:.1f}/100)."
        
        if issues:
            summary += f" Key concerns: {len(issues)} issue(s) identified."
        else:
            summary += " No major concerns identified."
        
        return summary
