"""
Investor Fit Scoring
Calculates how well the portfolio matches the investor's profile
"""

from typing import Dict
import numpy as np


class InvestorFitScorer:
    """Calculate Investor Fit Score (IFS) - portfolio suitability for specific investor"""
    
    def __init__(self):
        """Initialize investor fit scorer"""
        self.fit_diagnostics = []
        self.mismatch_details = []
    
    def calculate_portfolio_risk_index(self, portfolio_metrics: Dict) -> float:
        """
        Calculate Portfolio Risk Index (PRI) 0-100
        Higher = More risky portfolio
        
        Components:
        - Volatility
        - VaR
        - Max Drawdown
        - Beta
        """
        volatility = portfolio_metrics.get('volatility', 0.15)
        var_95 = portfolio_metrics.get('var_95', 0.15)
        max_dd = portfolio_metrics.get('max_drawdown', 0.10)
        beta = portfolio_metrics.get('beta', 1.0)
        
        # Normalize each component to 0-100 (higher = more risk)
        
        # Volatility score
        if volatility > 0.30:
            vol_score = 100
        elif volatility > 0.20:
            vol_score = 70 + (volatility - 0.20) * 300
        elif volatility > 0.15:
            vol_score = 50 + (volatility - 0.15) * 400
        elif volatility > 0.10:
            vol_score = 30 + (volatility - 0.10) * 400
        else:
            vol_score = volatility * 300
        
        # VaR score
        if var_95 > 0.30:
            var_score = 100
        elif var_95 > 0.20:
            var_score = 70 + (var_95 - 0.20) * 300
        elif var_95 > 0.15:
            var_score = 50 + (var_95 - 0.15) * 400
        else:
            var_score = var_95 * 333
        
        # Max Drawdown score
        if max_dd > 0.40:
            dd_score = 100
        elif max_dd > 0.25:
            dd_score = 70 + (max_dd - 0.25) * 200
        elif max_dd > 0.15:
            dd_score = 50 + (max_dd - 0.15) * 200
        else:
            dd_score = max_dd * 333
        
        # Beta score
        if beta > 1.5:
            beta_score = 100
        elif beta > 1.2:
            beta_score = 70 + (beta - 1.2) * 100
        elif beta > 1.0:
            beta_score = 50 + (beta - 1.0) * 100
        elif beta > 0.7:
            beta_score = 30 + (beta - 0.7) * 67
        else:
            beta_score = beta * 43
        
        # Weighted combination
        pri = (
            vol_score * 0.35 +
            var_score * 0.30 +
            dd_score * 0.25 +
            beta_score * 0.10
        )
        
        return min(100, max(0, pri))
    
    def calculate_portfolio_drawdown_severity(self, portfolio_metrics: Dict) -> float:
        """
        Calculate Portfolio Drawdown Severity (PDS) 0-100
        Measures behavioral pain potential
        
        Focus on downside risk and pain
        """
        max_dd = portfolio_metrics.get('max_drawdown', 0.10)
        var_99 = portfolio_metrics.get('var_99', 0.20)
        downside_deviation = portfolio_metrics.get('volatility', 0.15) * 0.7  # Approximate
        
        # Max Drawdown severity (40%)
        if max_dd > 0.40:
            dd_severity = 100
        elif max_dd > 0.30:
            dd_severity = 80 + (max_dd - 0.30) * 200
        elif max_dd > 0.20:
            dd_severity = 60 + (max_dd - 0.20) * 200
        elif max_dd > 0.15:
            dd_severity = 40 + (max_dd - 0.15) * 400
        else:
            dd_severity = max_dd * 267
        
        # VaR 99% severity (35%)
        if var_99 > 0.40:
            var_severity = 100
        elif var_99 > 0.30:
            var_severity = 80 + (var_99 - 0.30) * 200
        elif var_99 > 0.20:
            var_severity = 60 + (var_99 - 0.20) * 200
        else:
            var_severity = var_99 * 300
        
        # Downside deviation severity (25%)
        if downside_deviation > 0.25:
            down_severity = 100
        elif downside_deviation > 0.15:
            down_severity = 60 + (downside_deviation - 0.15) * 400
        else:
            down_severity = downside_deviation * 400
        
        pds = (
            dd_severity * 0.40 +
            var_severity * 0.35 +
            down_severity * 0.25
        )
        
        return min(100, max(0, pds))
    
    def calculate_ifs(self, 
                     investor_indices: Dict,
                     portfolio_metrics: Dict) -> Dict:
        """
        Calculate Investor Fit Score (IFS)
        
        Args:
            investor_indices: Dict with RCI, RTI, BFI, THS
            portfolio_metrics: Dict with portfolio risk metrics
            
        Returns:
            Dict with IFS score and diagnostics
        """
        # Extract investor indices
        rci = investor_indices.get('risk_capacity_index', 50)
        rti = investor_indices.get('risk_tolerance_index', 50)
        bfi = investor_indices.get('behavioral_fragility_index', 50)
        ths = investor_indices.get('time_horizon_strength', 50)
        
        # Calculate portfolio risk indices
        pri = self.calculate_portfolio_risk_index(portfolio_metrics)
        pds = self.calculate_portfolio_drawdown_severity(portfolio_metrics)
        
        # Calculate effective risk tolerance (fragility reduces usable tolerance)
        effective_rti = rti * (1 - bfi / 100)
        
        # Reset diagnostics
        self.fit_diagnostics = []
        self.mismatch_details = []
        
        # Start with perfect score
        ifs = 100
        
        # Weights for different mismatch types
        w1 = 0.4  # Structural capacity mismatch
        w2 = 0.4  # Behavioral tolerance mismatch
        w3 = 0.2  # Fragility amplifier
        
        # Rule 1: Capacity >= Portfolio Risk
        capacity_mismatch = max(0, pri - rci)
        if capacity_mismatch > 0:
            penalty = w1 * capacity_mismatch
            ifs -= penalty
            self.mismatch_details.append({
                'type': 'capacity_mismatch',
                'severity': capacity_mismatch,
                'penalty': penalty,
                'message': f"Portfolio risk ({pri:.0f}) exceeds your risk capacity ({rci:.0f})"
            })
        
        # Rule 2: Effective Tolerance >= Drawdown Severity
        tolerance_mismatch = max(0, pds - effective_rti)
        if tolerance_mismatch > 0:
            penalty = w2 * tolerance_mismatch
            ifs -= penalty
            self.mismatch_details.append({
                'type': 'tolerance_mismatch',
                'severity': tolerance_mismatch,
                'penalty': penalty,
                'message': f"Portfolio drawdown risk ({pds:.0f}) exceeds your tolerance ({effective_rti:.0f})"
            })
        
        # Rule 3: Fragility Amplifier
        fragility_penalty = w3 * (bfi * pri / 100)
        ifs -= fragility_penalty
        if bfi > 40:  # High fragility
            self.mismatch_details.append({
                'type': 'behavioral_fragility',
                'severity': bfi,
                'penalty': fragility_penalty,
                'message': f"High behavioral fragility ({bfi:.0f}) amplifies portfolio risk"
            })
        
        # Ensure IFS is in valid range
        ifs = max(0, min(100, ifs))
        
        # Generate diagnostics
        self._generate_diagnostics(
            ifs, rci, rti, bfi, ths, pri, pds, effective_rti,
            capacity_mismatch, tolerance_mismatch
        )
        
        # Determine fit category
        if ifs >= 80:
            category = "Excellent Fit"
        elif ifs >= 65:
            category = "Good Fit"
        elif ifs >= 50:
            category = "Acceptable Fit"
        elif ifs >= 35:
            category = "Poor Fit"
        else:
            category = "Very Poor Fit"
        
        return {
            'ifs_score': round(ifs, 1),
            'category': category,
            'investor_indices': {
                'risk_capacity': round(rci, 1),
                'risk_tolerance': round(rti, 1),
                'effective_tolerance': round(effective_rti, 1),
                'behavioral_fragility': round(bfi, 1),
                'time_horizon_strength': round(ths, 1)
            },
            'portfolio_indices': {
                'risk_index': round(pri, 1),
                'drawdown_severity': round(pds, 1)
            },
            'mismatches': self.mismatch_details,
            'diagnostics': self.fit_diagnostics,
            'interpretation': self._get_interpretation(ifs)
        }
    
    def _generate_diagnostics(self, ifs, rci, rti, bfi, ths, pri, pds, 
                             effective_rti, capacity_mismatch, tolerance_mismatch):
        """Generate detailed fit diagnostics"""
        
        # Capacity-based diagnostics
        if capacity_mismatch > 30:
            self.fit_diagnostics.append(
                f"❌ CRITICAL: Portfolio risk significantly exceeds your financial capacity"
            )
        elif capacity_mismatch > 15:
            self.fit_diagnostics.append(
                f"⚠️ WARNING: Portfolio risk moderately exceeds your capacity"
            )
        elif capacity_mismatch > 0:
            self.fit_diagnostics.append(
                f"⚠️ CAUTION: Portfolio risk slightly exceeds your capacity"
            )
        else:
            self.fit_diagnostics.append(
                f"✅ Portfolio risk is within your financial capacity"
            )
        
        # Tolerance-based diagnostics
        if tolerance_mismatch > 30:
            self.fit_diagnostics.append(
                f"❌ CRITICAL: Portfolio drawdown potential ({pds:.0f}) far exceeds "
                f"your psychological comfort ({effective_rti:.0f})"
            )
        elif tolerance_mismatch > 15:
            self.fit_diagnostics.append(
                f"⚠️ WARNING: Portfolio drawdown risk exceeds your tolerance threshold"
            )
        elif tolerance_mismatch > 0:
            self.fit_diagnostics.append(
                f"⚠️ CAUTION: Portfolio may cause psychological discomfort during downturns"
            )
        else:
            self.fit_diagnostics.append(
                f"✅ Portfolio drawdown risk is within your tolerance"
            )
        
        # Fragility diagnostics
        if bfi > 60:
            self.fit_diagnostics.append(
                f"❌ High behavioral fragility ({bfi:.0f}/100) increases abandonment risk"
            )
        elif bfi > 40:
            self.fit_diagnostics.append(
                f"⚠️ Moderate behavioral fragility may affect plan adherence"
            )
        else:
            self.fit_diagnostics.append(
                f"✅ Strong behavioral discipline indicates good plan adherence"
            )
        
        # Time horizon diagnostics
        if ths < 40:
            self.fit_diagnostics.append(
                f"⚠️ Short/rigid time horizon limits portfolio flexibility"
            )
        elif ths >= 70:
            self.fit_diagnostics.append(
                f"✅ Strong time horizon supports long-term strategy"
            )
    
    def _get_interpretation(self, ifs: float) -> str:
        """Get human-readable interpretation of IFS"""
        if ifs >= 80:
            return "This portfolio is an excellent match for your profile and risk capacity."
        elif ifs >= 65:
            return "This portfolio is generally suitable for you with minor concerns."
        elif ifs >= 50:
            return "This portfolio has acceptable fit but shows some misalignment with your profile."
        elif ifs >= 35:
            return "This portfolio is poorly suited to your risk profile and capacity."
        else:
            return "This portfolio has severe mismatches with your profile and should be reconsidered."
    
    def calculate_plan_survival_probability(self,
                                           ifs: float,
                                           bfi: float,
                                           ths: float) -> Dict:
        """
        Calculate Plan Survival Probability (PSP)
        Probability investor stays invested for full horizon
        
        Args:
            ifs: Investor Fit Score
            bfi: Behavioral Fragility Index
            ths: Time Horizon Strength
            
        Returns:
            Dict with PSP and interpretation
        """
        # Base survival probability from fit
        base_psp = ifs
        
        # Behavioral fragility reduces survival
        fragility_impact = -bfi * 0.5  # High fragility = high abandonment
        
        # Time horizon strength increases survival
        horizon_boost = (ths - 50) * 0.3
        
        # Calculate PSP
        psp = base_psp + fragility_impact + horizon_boost
        psp = max(0, min(100, psp))
        
        # Interpretation
        if psp >= 80:
            interpretation = "Very high probability of completing investment plan"
            risk_level = "Low abandonment risk"
        elif psp >= 60:
            interpretation = "Good probability of plan completion with discipline"
            risk_level = "Moderate abandonment risk"
        elif psp >= 40:
            interpretation = "Moderate probability - requires behavioral support"
            risk_level = "High abandonment risk"
        else:
            interpretation = "Low probability - high risk of plan abandonment"
            risk_level = "Very high abandonment risk"
        
        return {
            'psp_score': round(psp, 1),
            'interpretation': interpretation,
            'risk_level': risk_level,
            'components': {
                'base_fit': round(base_psp, 1),
                'fragility_impact': round(fragility_impact, 1),
                'horizon_boost': round(horizon_boost, 1)
            }
        }
