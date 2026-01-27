"""
Portfolio Health Analysis System
Main script to run complete portfolio analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from portfolio_metrics import PortfolioMetrics
from benchmark_analyzer import BenchmarkAnalyzer
from objective_alignment import ObjectiveAlignmentAnalyzer, InvestmentObjective
from portfolio_health import PortfolioHealthClassifier
from investor_profile import InvestorProfileAnalyzer
from portfolio_quality import PortfolioQualityScorer
from investor_fit import InvestorFitScorer


class PortfolioAnalyzer:
    """Main portfolio analysis orchestrator"""
    
    def __init__(self, portfolio_path: str, iid_path: str = None, 
                 investment_objective: str = "Moderate Growth"):
        """
        Initialize portfolio analyzer
        
        Args:
            portfolio_path: Path to Excel file with portfolio data
            iid_path: Path to IID JSON file (optional)
            investment_objective: Investment objective (Conservative/Moderate/Aggressive/Balanced)
        """
        self.portfolio_path = portfolio_path
        self.iid_path = iid_path
        self.investment_objective = investment_objective
        
        # Initialize components
        self.data_fetcher = DataFetcher(period_years=1)
        self.metrics_calc = PortfolioMetrics(risk_free_rate=self.data_fetcher.get_risk_free_rate())
        self.benchmark_analyzer = BenchmarkAnalyzer(self.data_fetcher, self.metrics_calc)
        self.objective_analyzer = ObjectiveAlignmentAnalyzer()
        self.health_classifier = PortfolioHealthClassifier()
        
        # New components for two-dimensional scoring
        self.investor_analyzer = InvestorProfileAnalyzer(iid_path) if iid_path else None
        self.quality_scorer = PortfolioQualityScorer()
        self.fit_scorer = InvestorFitScorer()
        
        # Data containers
        self.portfolio_df = None
        self.holdings_data = {}
        self.portfolio_returns = None
        self.metrics = {}
        
    def load_portfolio(self):
        """Load portfolio from Excel file"""
        print("=" * 80)
        print("LOADING PORTFOLIO DATA")
        print("=" * 80)
        
        print(f"Attempting to load: {self.portfolio_path}")
        
        # Get all available sheet names
        try:
            xl_file = pd.ExcelFile(self.portfolio_path)
            available_sheets = xl_file.sheet_names
            print(f"Available sheets: {available_sheets}")
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            raise
        
        # Try multiple sheet names for compatibility
        sheet_names_to_try = [
            'Portfolio Data_Hypothetical',
            'Portfolio Data',
            'Portfolio',
            'Holdings',
            'holdings (3)',  # Common default name
            None  # Use first sheet if specific names not found
        ]
        
        self.portfolio_df = None
        loaded_sheet = None
        
        for sheet_name in sheet_names_to_try:
            try:
                if sheet_name is None and len(available_sheets) > 0:
                    # Use the first available sheet
                    sheet_name = available_sheets[0]
                    self.portfolio_df = pd.read_excel(self.portfolio_path, sheet_name=sheet_name)
                    loaded_sheet = sheet_name
                    print(f"✓ Loaded first sheet: '{sheet_name}'")
                elif sheet_name in available_sheets:
                    self.portfolio_df = pd.read_excel(self.portfolio_path, sheet_name=sheet_name)
                    loaded_sheet = sheet_name
                    print(f"✓ Loaded sheet: '{sheet_name}'")
                else:
                    continue
                break
            except Exception as e:
                print(f"Failed to load sheet '{sheet_name}': {e}")
                continue
        
        if self.portfolio_df is None:
            raise ValueError(f"Could not load any compatible sheet from {self.portfolio_path}. Available sheets: {available_sheets}")
        
        # Add missing 'Sector' column if it doesn't exist
        if 'Sector' not in self.portfolio_df.columns:
            print("⚠️ 'Sector' column not found. Adding default sector 'Unclassified'")
            self.portfolio_df['Sector'] = 'Unclassified'
        
        # Add missing 'Asset Class' column if it doesn't exist
        if 'Asset Class' not in self.portfolio_df.columns:
            print("⚠️ 'Asset Class' column not found. Adding default asset class 'Equity'")
            self.portfolio_df['Asset Class'] = 'Equity'
        
        print(f"\nPortfolio loaded from sheet '{loaded_sheet}': {len(self.portfolio_df)} holdings")
        print(f"Total value: ₹{self.portfolio_df['Cur. val'].sum():,.2f}")
        print(f"\nHoldings:")
        for idx, row in self.portfolio_df.iterrows():
            print(f"  - {row['Instrument']}: ₹{row['Cur. val']:,.2f} ({row['Sector']})")
        
        return self.portfolio_df
    
    def fetch_historical_data(self):
        """Fetch historical price data for all holdings"""
        print("\n" + "=" * 80)
        print("FETCHING HISTORICAL DATA")
        print("=" * 80)
        
        instruments = self.portfolio_df['Instrument'].tolist()
        self.holdings_data = self.data_fetcher.fetch_portfolio_data(instruments)
        
        # Check what we got
        successful = sum(1 for v in self.holdings_data.values() if v is not None)
        print(f"\nSuccessfully fetched data for {successful}/{len(instruments)} holdings")
        
        return self.holdings_data
    
    def calculate_portfolio_returns(self):
        """Calculate weighted portfolio returns"""
        print("\n" + "=" * 80)
        print("CALCULATING PORTFOLIO RETURNS")
        print("=" * 80)
        
        total_value = self.portfolio_df['Cur. val'].sum()
        
        # Calculate weighted returns for each holding
        all_returns = []
        weights = []
        
        for idx, row in self.portfolio_df.iterrows():
            instrument = row['Instrument']
            weight = row['Cur. val'] / total_value
            
            if instrument in self.holdings_data and self.holdings_data[instrument] is not None:
                returns = self.data_fetcher.calculate_returns(self.holdings_data[instrument])
                if returns is not None and len(returns) > 0:
                    all_returns.append(returns)
                    weights.append(weight)
                    print(f"  ✓ {instrument}: {weight*100:.1f}% weight")
        
        if not all_returns:
            print("\n⚠️ Warning: No return data available")
            return None
        
        # Align all return series
        returns_df = pd.DataFrame({i: ret for i, ret in enumerate(all_returns)})
        returns_df = returns_df.fillna(0)  # Fill missing with 0
        
        # Calculate weighted portfolio returns
        weights_array = np.array(weights) / sum(weights)  # Normalize weights
        self.portfolio_returns = returns_df.dot(weights_array)
        
        print(f"\nPortfolio returns calculated: {len(self.portfolio_returns)} days of data")
        
        return self.portfolio_returns
    
    def calculate_metrics(self) -> Dict:
        """Calculate all portfolio metrics"""
        print("\n" + "=" * 80)
        print("CALCULATING PORTFOLIO METRICS")
        print("=" * 80)
        
        if self.portfolio_returns is None or len(self.portfolio_returns) == 0:
            print("⚠️ Cannot calculate metrics without return data")
            return {}
        
        # Get primary benchmark for this portfolio
        primary_benchmark = self.benchmark_analyzer.determine_portfolio_benchmark(self.portfolio_df)
        benchmark_data = self.data_fetcher.fetch_benchmark_data(primary_benchmark)
        benchmark_returns = self.data_fetcher.calculate_returns(benchmark_data) if benchmark_data is not None else None
        
        # Calculate all metrics
        metrics = self.metrics_calc.calculate_all_metrics(self.portfolio_returns, benchmark_returns)
        
        # Display metrics
        print("\n📊 PERFORMANCE METRICS:")
        print(f"  Annual Return:       {metrics['annual_return']:.2f}%" if metrics['annual_return'] else "  Annual Return:       N/A")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}" if metrics['sharpe_ratio'] else "  Sharpe Ratio:        N/A")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:.3f}" if metrics['sortino_ratio'] else "  Sortino Ratio:       N/A")
        
        print("\n⚠️ RISK METRICS:")
        print(f"  VaR (95%):           {metrics['var_95']:.2f}%" if metrics['var_95'] else "  VaR (95%):           N/A")
        print(f"  VaR (99%):           {metrics['var_99']:.2f}%" if metrics['var_99'] else "  VaR (99%):           N/A")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:.2f}%" if metrics['max_drawdown'] else "  Max Drawdown:        N/A")
        print(f"  Volatility:          {metrics['volatility']:.2f}%" if metrics['volatility'] else "  Volatility:          N/A")
        
        if metrics.get('jensens_alpha'):
            print("\n📈 BENCHMARK-RELATIVE:")
            print(f"  Jensen's Alpha:      {metrics['jensens_alpha']:.2f}%")
            print(f"  Beta:                {metrics['beta']:.3f}" if metrics.get('beta') else "  Beta:                N/A")
        
        return metrics
    
    def analyze_benchmarks(self, metrics: Dict) -> Dict:
        """Analyze performance against benchmarks"""
        print("\n" + "=" * 80)
        print("BENCHMARK ANALYSIS")
        print("=" * 80)
        
        benchmark_summary = self.benchmark_analyzer.generate_benchmark_summary(
            self.portfolio_returns, 
            self.portfolio_df
        )
        
        primary = benchmark_summary['primary_benchmark']
        print(f"\n🎯 Primary Benchmark: {primary['benchmark_name']}")
        print(f"  Portfolio Return:     {primary['portfolio_return']:.2f}%" if primary['portfolio_return'] else "  Portfolio Return:     N/A")
        print(f"  Benchmark Return:     {primary['benchmark_return']:.2f}%" if primary['benchmark_return'] else "  Benchmark Return:     N/A")
        
        if primary['outperformance'] is not None:
            symbol = "📈" if primary['outperformance'] > 0 else "📉"
            print(f"  Outperformance:       {symbol} {primary['outperformance']:+.2f}%")
        
        if benchmark_summary['nifty_50_comparison']:
            nifty = benchmark_summary['nifty_50_comparison']
            print(f"\n📊 Nifty 50 Comparison:")
            print(f"  Outperformance:       {nifty['outperformance']:+.2f}%" if nifty['outperformance'] else "  Outperformance:       N/A")
        
        print(f"\n{benchmark_summary['summary']}")
        
        return benchmark_summary
    
    def analyze_objective_alignment(self, metrics: Dict) -> Dict:
        """Analyze alignment with investment objective"""
        print("\n" + "=" * 80)
        print("OBJECTIVE ALIGNMENT ANALYSIS")
        print("=" * 80)
        
        objective = self.objective_analyzer.parse_objective(self.investment_objective)
        alignment = self.objective_analyzer.evaluate_alignment(metrics, objective)
        
        print(f"\n🎯 Investment Objective: {alignment['objective']}")
        print(f"📊 Alignment Score: {alignment['overall_score']:.1f}/100")
        print(f"🏷️ Category: {alignment['alignment_category']}")
        
        print("\n📋 Individual Scores:")
        for criterion, score in alignment['individual_scores'].items():
            print(f"  {criterion.capitalize():15} {score:.1f}/100")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(alignment['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        return alignment
    
    def classify_health(self, 
                       metrics: Dict,
                       benchmark_summary: Dict,
                       objective_alignment: Dict) -> Dict:
        """Classify overall portfolio health"""
        print("\n" + "=" * 80)
        print("PORTFOLIO HEALTH CLASSIFICATION")
        print("=" * 80)
        
        health = self.health_classifier.classify_health(
            metrics,
            benchmark_summary['primary_benchmark'],
            objective_alignment
        )
        
        # Display health status
        status_emoji = {
            "Healthy": "✅",
            "Warning": "⚠️",
            "At Risk": "🚨",
            "Critical": "🔴"
        }
        
        emoji = status_emoji.get(health['health_status'], "❓")
        print(f"\n{emoji} HEALTH STATUS: {health['health_status']}")
        print(f"Overall Score: {health['overall_health_score']:.1f}/100")
        
        print("\n📊 Component Scores:")
        for component, score in health['component_scores'].items():
            print(f"  {component.replace('_', ' ').title():20} {score:.1f}/100")
        
        if health['key_issues']:
            print("\n⚠️ Key Issues:")
            for i, issue in enumerate(health['key_issues'], 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✅ No major issues identified")
        
        print("\n📝 Action Items:")
        for i, action in enumerate(health['action_items'], 1):
            print(f"  {i}. {action}")
        
        print(f"\n📋 Summary:")
        print(f"  {health['summary']}")
        
        return health
    
    def run_complete_analysis(self) -> Dict:
        """Run complete portfolio analysis pipeline"""
        print("\n" + "=" * 80)
        print("PORTFOLIO HEALTH ANALYSIS SYSTEM")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Step 1: Load portfolio
            self.load_portfolio()
            
            # Step 2: Fetch historical data
            self.fetch_historical_data()
            
            # Step 3: Calculate portfolio returns
            self.calculate_portfolio_returns()
            
            # Step 4: Calculate metrics
            metrics = self.calculate_metrics()
            self.metrics = metrics  # Store for later use
            
            # Step 5: Benchmark analysis
            benchmark_summary = self.analyze_benchmarks(metrics)
            
            # Step 6: Objective alignment
            objective_alignment = self.analyze_objective_alignment(metrics)
            
            # Step 7: Health classification
            health = self.classify_health(metrics, benchmark_summary, objective_alignment)
            
            # NEW: Step 8: Two-Dimensional Scoring
            pqs_result = None
            ifs_result = None
            investor_summary = None
            psp_result = None
            
            if self.investor_analyzer:
                # Calculate Portfolio Quality Score
                pqs_result = self.calculate_portfolio_quality_score(metrics)
                
                # Get investor profile
                investor_summary = self.investor_analyzer.get_investor_summary()
                
                # Calculate Investor Fit Score
                ifs_result = self.calculate_investor_fit_score(
                    investor_summary['indices'],
                    metrics
                )
                
                # Calculate Plan Survival Probability
                psp_result = self.fit_scorer.calculate_plan_survival_probability(
                    ifs_result['ifs_score'],
                    investor_summary['indices']['behavioral_fragility_index'],
                    investor_summary['indices']['time_horizon_strength']
                )
                
                # Display two-dimensional analysis
                self.display_two_dimensional_analysis(pqs_result, ifs_result, psp_result)
            
            # Compile complete report
            report = {
                'portfolio': self.portfolio_df.to_dict('records'),
                'metrics': metrics,
                'benchmark_analysis': benchmark_summary,
                'objective_alignment': objective_alignment,
                'health_classification': health,
                'portfolio_quality_score': pqs_result,
                'investor_fit_score': ifs_result,
                'investor_profile': investor_summary,
                'plan_survival_probability': psp_result,
                'analysis_date': datetime.now().isoformat()
            }
            
            print("\n" + "=" * 80)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 80)
            
            return report
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def calculate_portfolio_quality_score(self, metrics: Dict) -> Dict:
        """Calculate Portfolio Quality Score (PQS)"""
        print("\n" + "=" * 80)
        print("PORTFOLIO QUALITY SCORE (PQS)")
        print("=" * 80)
        
        pqs_result = self.quality_scorer.calculate_pqs(metrics)
        
        print(f"\n🎯 Portfolio Quality Score: {pqs_result['pqs_score']}/100")
        print(f"   Category: {pqs_result['category']}")
        print(f"\n   {pqs_result['interpretation']}")
        
        print(f"\n📊 Component Scores:")
        for component, score in pqs_result['components'].items():
            weight = pqs_result['weights'][component]
            print(f"   {component.replace('_', ' ').title():20} {score:>5.1f}/100  (weight: {weight}%)")
        
        return pqs_result
    
    def calculate_investor_fit_score(self, investor_indices: Dict, 
                                     portfolio_metrics: Dict) -> Dict:
        """Calculate Investor Fit Score (IFS)"""
        print("\n" + "=" * 80)
        print("INVESTOR FIT SCORE (IFS)")
        print("=" * 80)
        
        # Display investor indices first
        print(f"\n👤 Investor Profile Indices:")
        print(f"   Risk Capacity Index:        {investor_indices['risk_capacity_index']:.1f}/100")
        print(f"   Risk Tolerance Index:       {investor_indices['risk_tolerance_index']:.1f}/100")
        print(f"   Behavioral Fragility Index: {investor_indices['behavioral_fragility_index']:.1f}/100")
        print(f"   Time Horizon Strength:      {investor_indices['time_horizon_strength']:.1f}/100")
        
        ifs_result = self.fit_scorer.calculate_ifs(investor_indices, portfolio_metrics)
        
        print(f"\n🎯 Investor Fit Score: {ifs_result['ifs_score']}/100")
        print(f"   Category: {ifs_result['category']}")
        print(f"\n   {ifs_result['interpretation']}")
        
        print(f"\n📊 Portfolio Risk Indices:")
        print(f"   Portfolio Risk Index:       {ifs_result['portfolio_indices']['risk_index']:.1f}/100")
        print(f"   Drawdown Severity:          {ifs_result['portfolio_indices']['drawdown_severity']:.1f}/100")
        
        if ifs_result['mismatches']:
            print(f"\n⚠️ Identified Mismatches:")
            for mismatch in ifs_result['mismatches']:
                print(f"   • {mismatch['message']}")
                print(f"     Severity: {mismatch['severity']:.1f}, Penalty: {mismatch['penalty']:.1f}")
        
        print(f"\n📋 Fit Diagnostics:")
        for diagnostic in ifs_result['diagnostics']:
            print(f"   {diagnostic}")
        
        return ifs_result
    
    def display_two_dimensional_analysis(self, pqs_result: Dict, 
                                        ifs_result: Dict, 
                                        psp_result: Dict):
        """Display the two-dimensional analysis summary"""
        print("\n" + "=" * 80)
        print("TWO-DIMENSIONAL PORTFOLIO ANALYSIS")
        print("=" * 80)
        
        pqs = pqs_result['pqs_score']
        ifs = ifs_result['ifs_score']
        
        print(f"\n┌{'─' * 78}┐")
        print(f"│ {'PORTFOLIO QUALITY SCORE (PQS)':^38} │ {'INVESTOR FIT SCORE (IFS)':^38} │")
        print(f"│ {pqs:>38.1f}/100 │ {ifs:>38.1f}/100 │")
        print(f"│ {pqs_result['category']:^38} │ {ifs_result['category']:^38} │")
        print(f"└{'─' * 78}┘")
        
        # Quadrant analysis
        print(f"\n📍 Portfolio Positioning:")
        
        if pqs >= 65 and ifs >= 65:
            quadrant = "✅ IDEAL ZONE"
            message = "Excellent portfolio that is well-suited for you"
        elif pqs >= 65 and ifs < 65:
            quadrant = "⚠️ GOOD PORTFOLIO, POOR FIT"
            message = "Strong portfolio performance, but misaligned with your profile"
        elif pqs < 65 and ifs >= 65:
            quadrant = "⚠️ POOR PORTFOLIO, GOOD FIT"
            message = "Portfolio matches your profile but has performance issues"
        else:
            quadrant = "❌ NEEDS IMPROVEMENT"
            message = "Portfolio needs both performance improvement and better alignment"
        
        print(f"   {quadrant}")
        print(f"   {message}")
        
        # Plan Survival Probability
        if psp_result:
            print(f"\n⭐ Plan Survival Probability: {psp_result['psp_score']:.1f}/100")
            print(f"   {psp_result['interpretation']}")
            print(f"   Risk Level: {psp_result['risk_level']}")
        
        # Strategic recommendations
        print(f"\n💡 Strategic Recommendations:")
        
        if pqs < 50:
            print(f"   1. IMPROVE PORTFOLIO QUALITY:")
            print(f"      - Review underperforming holdings")
            print(f"      - Optimize for better risk-adjusted returns")
            print(f"      - Consider rebalancing strategy")
        
        if ifs < 50:
            print(f"   2. IMPROVE INVESTOR FIT:")
            if ifs_result['mismatches']:
                for mismatch in ifs_result['mismatches']:
                    if mismatch['severity'] > 20:
                        if 'capacity' in mismatch['type']:
                            print(f"      - Reduce portfolio risk to match your financial capacity")
                        elif 'tolerance' in mismatch['type']:
                            print(f"      - Lower exposure to high-volatility assets")
                        elif 'fragility' in mismatch['type']:
                            print(f"      - Add behavioral guardrails (SIPs, auto-rebalancing)")
        
        if ifs < 65 and pqs >= 65:
            print(f"   3. PORTFOLIO IS GOOD BUT NOT FOR YOU:")
            print(f"      - Consider this a temporary mismatch")
            print(f"      - Either adjust portfolio risk OR")
            print(f"      - Build capacity through financial planning")


def main():
    """Main entry point"""
    # Configuration
    PORTFOLIO_PATH = r'f:\Insights\Portfolio Data_Hypothetical.xlsx'
    IID_PATH = r'F:\AI Insights Dashboard\IID SON.txt'  # Path to IID file
    INVESTMENT_OBJECTIVE = "Moderate Growth"  # Change as needed
    
    # Create analyzer with IID
    analyzer = PortfolioAnalyzer(
        portfolio_path=PORTFOLIO_PATH,
        iid_path=IID_PATH,
        investment_objective=INVESTMENT_OBJECTIVE
    )
    
    # Run analysis
    report = analyzer.run_complete_analysis()
    
    return report


if __name__ == "__main__":
    report = main()
