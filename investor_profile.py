"""
Investor Profile Analyzer
Extracts key investor signals from IID (Investor Information Document)
"""

import json
from typing import Dict, Tuple
from pathlib import Path


class InvestorProfileAnalyzer:
    """Analyze investor profile and derive risk indices from IID"""
    
    def __init__(self, iid_path: str):
        """
        Initialize with IID file path
        
        Args:
            iid_path: Path to IID JSON file
        """
        self.iid_path = iid_path
        self.iid_data = self._load_iid()
        
        # Calculated indices
        self.rci = 0  # Risk Capacity Index
        self.rti = 0  # Risk Tolerance Index
        self.bfi = 0  # Behavioral Fragility Index
        self.ths = 0  # Time Horizon Strength
        
    def _load_iid(self) -> Dict:
        """Load IID JSON file"""
        try:
            with open(self.iid_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading IID file: {str(e)}")
            return {}
    
    def calculate_risk_capacity_index(self) -> float:
        """
        Calculate Risk Capacity Index (0-100)
        Measures ABILITY to bear risk
        
        Components:
        - Income stability (30%)
        - Emergency fund (25%)
        - Liability burden (25%)
        - Portfolio concentration (20%)
        """
        score = 0
        risk_profile = self.iid_data.get('risk_profile', {})
        capacity = risk_profile.get('risk_capacity', {})
        family = self.iid_data.get('family_and_dependents', {})
        
        # 1. Income Stability (30 points)
        income_stability = capacity.get('income_stability', 'stable')
        stability_scores = {
            'very_stable': 30,
            'stable': 22,
            'variable': 12,
            'volatile': 5
        }
        score += stability_scores.get(income_stability, 15)
        
        # 2. Emergency Fund (25 points)
        emergency_months = capacity.get('emergency_fund_months', 0)
        if emergency_months >= 12:
            score += 25
        elif emergency_months >= 6:
            score += 20
        elif emergency_months >= 3:
            score += 12
        else:
            score += 5
        
        # 3. Liability Burden (25 points)
        liabilities = capacity.get('existing_liabilities', {})
        has_liabilities = liabilities.get('has_liabilities', False)
        
        if not has_liabilities:
            score += 25
        else:
            # Lower score if EMI exists (proxy for burden)
            emi = liabilities.get('monthly_emi_amount', 0)
            if emi < 20000:
                score += 20
            elif emi < 40000:
                score += 15
            elif emi < 60000:
                score += 8
            else:
                score += 3
        
        # 4. Portfolio Concentration (20 points)
        portfolio_pct = capacity.get('portfolio_as_percent_of_net_worth', 50)
        if portfolio_pct < 30:
            score += 20  # Well diversified
        elif portfolio_pct < 50:
            score += 15
        elif portfolio_pct < 70:
            score += 8
        else:
            score += 3  # Too concentrated
        
        # 5. Dependents penalty
        num_dependents = family.get('number_of_dependents', 0)
        if num_dependents > 3:
            score -= 10
        elif num_dependents > 1:
            score -= 5
        
        self.rci = max(0, min(100, score))
        return self.rci
    
    def calculate_risk_tolerance_index(self) -> float:
        """
        Calculate Risk Tolerance Index (0-100)
        Measures WILLINGNESS to accept risk
        
        Components:
        - Drawdown response (40%)
        - Sleep-loss threshold (35%)
        - Loss vs gain preference (25%)
        """
        score = 0
        risk_profile = self.iid_data.get('risk_profile', {})
        tolerance = risk_profile.get('risk_tolerance', {})
        
        # 1. Drawdown Response (40 points)
        drawdown_response = tolerance.get('drawdown_response', 'hold')
        response_scores = {
            'invest_more': 40,  # Contrarian/brave
            'hold': 30,         # Disciplined
            'reduce': 15,       # Nervous
            'exit': 5           # Panic-prone
        }
        score += response_scores.get(drawdown_response, 20)
        
        # 2. Sleep-loss Drawdown Threshold (35 points)
        sleep_loss_pct = tolerance.get('sleep_loss_drawdown_percent', 20)
        if sleep_loss_pct >= 40:
            score += 35  # Very high tolerance
        elif sleep_loss_pct >= 30:
            score += 28
        elif sleep_loss_pct >= 20:
            score += 20
        elif sleep_loss_pct >= 15:
            score += 12
        else:
            score += 5  # Very low tolerance
        
        # 3. Loss vs Missed Gains (25 points)
        loss_preference = tolerance.get('loss_vs_missed_gains', 'loss_averse')
        if loss_preference == 'gain_seeking':
            score += 25
        else:
            score += 10  # Loss averse = lower risk tolerance
        
        self.rti = max(0, min(100, score))
        return self.rti
    
    def calculate_behavioral_fragility_index(self) -> float:
        """
        Calculate Behavioral Fragility Index (0-100)
        Higher = More likely to abandon plan / panic
        
        Components:
        - Past exit behavior (40%)
        - Number of abandonment triggers (30%)
        - Decision autonomy (30%)
        """
        score = 0
        risk_profile = self.iid_data.get('risk_profile', {})
        behavior_history = risk_profile.get('risk_behavior_history', {})
        triggers = self.iid_data.get('behavioral_triggers', {})
        
        # 1. Past Exit Behavior (40 points)
        exited_before = behavior_history.get('exited_during_crash_before', False)
        reentered = behavior_history.get('reentered_after_exit', True)
        
        if exited_before and not reentered:
            score += 40  # Very fragile - exited and never came back
        elif exited_before and reentered:
            score += 25  # Moderate fragility - exited but learned
        else:
            score += 5   # Low fragility - stayed invested
        
        # 2. Abandonment Triggers (30 points)
        abandonment_triggers = triggers.get('plan_abandonment_triggers', [])
        trigger_count = len(abandonment_triggers)
        
        if trigger_count >= 4:
            score += 30  # Many triggers = high fragility
        elif trigger_count >= 3:
            score += 22
        elif trigger_count >= 2:
            score += 15
        elif trigger_count >= 1:
            score += 8
        else:
            score += 0  # No triggers = robust
        
        # 3. Decision Autonomy (30 points)
        autonomy = triggers.get('decision_autonomy', 'guided')
        autonomy_scores = {
            'full_control': 30,    # High fragility - no rules
            'guided': 15,          # Medium - some structure
            'rules_based': 5       # Low fragility - disciplined
        }
        score += autonomy_scores.get(autonomy, 15)
        
        self.bfi = max(0, min(100, score))
        return self.bfi
    
    def calculate_time_horizon_strength(self) -> float:
        """
        Calculate Time Horizon Strength (0-100)
        Measures ability to stay invested long-term
        
        Components:
        - Horizon duration (40%)
        - Horizon flexibility (35%)
        - Dependency timeline (25%)
        """
        score = 0
        horizon = self.iid_data.get('time_horizon', {})
        family = self.iid_data.get('family_and_dependents', {})
        
        # 1. Horizon Duration (40 points)
        earliest_year = horizon.get('earliest_liquidity_year', 2030)
        comfortable_year = horizon.get('comfortable_exit_year', 2035)
        current_year = 2025
        
        earliest_duration = earliest_year - current_year
        comfortable_duration = comfortable_year - current_year
        
        if comfortable_duration >= 20:
            score += 40
        elif comfortable_duration >= 15:
            score += 35
        elif comfortable_duration >= 10:
            score += 28
        elif comfortable_duration >= 7:
            score += 20
        elif comfortable_duration >= 5:
            score += 12
        else:
            score += 5
        
        # 2. Horizon Flexibility (35 points)
        horizon_type = horizon.get('horizon_type', 'fixed')
        type_scores = {
            'flexible': 35,      # Can extend if needed
            'aspirational': 25,  # Some flexibility
            'fixed': 10          # Hard deadline
        }
        score += type_scores.get(horizon_type, 20)
        
        # 3. Dependency Timeline (25 points)
        dependency_years = family.get('dependency_duration_years', 10)
        
        # If dependencies end before comfortable exit = good
        if dependency_years < comfortable_duration - 5:
            score += 25
        elif dependency_years < comfortable_duration:
            score += 18
        elif dependency_years == comfortable_duration:
            score += 10
        else:
            score += 3  # Dependencies extend beyond horizon
        
        self.ths = max(0, min(100, score))
        return self.ths
    
    def get_all_indices(self) -> Dict[str, float]:
        """Calculate and return all investor indices"""
        return {
            'risk_capacity_index': self.calculate_risk_capacity_index(),
            'risk_tolerance_index': self.calculate_risk_tolerance_index(),
            'behavioral_fragility_index': self.calculate_behavioral_fragility_index(),
            'time_horizon_strength': self.calculate_time_horizon_strength()
        }
    
    def get_investor_summary(self) -> Dict:
        """Get comprehensive investor summary"""
        indices = self.get_all_indices()
        
        profile = self.iid_data.get('investor_profile', {})
        motivation = self.iid_data.get('investment_motivation', {})
        horizon = self.iid_data.get('time_horizon', {})
        
        return {
            'investor_id': self.iid_data.get('investor_id', 'Unknown'),
            'age_band': profile.get('age_band', 'Unknown'),
            'employment_type': profile.get('employment_type', 'Unknown'),
            'primary_goal': motivation.get('goal_priority_rank', {}).get('primary', 'Unknown'),
            'time_horizon_years': horizon.get('comfortable_exit_year', 2030) - 2025,
            'indices': indices,
            'effective_risk_tolerance': indices['risk_tolerance_index'] * (1 - indices['behavioral_fragility_index'] / 100)
        }
