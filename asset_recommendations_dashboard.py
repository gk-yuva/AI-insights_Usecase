"""
Asset Recommendations Dashboard
Streamlit app for recommending assets to add/drop from Nifty50
Standalone dashboard separate from the main portfolio health dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import plotly.graph_objects as go

from portfolio_optimizer import PortfolioOptimizer
from data_fetcher import DataFetcher
from portfolio_metrics import PortfolioMetrics

# Page configuration
st.set_page_config(
    page_title="Asset Recommendations",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


def load_portfolio_from_file(filepath):
    """Load portfolio from Excel file"""
    try:
        df = pd.read_excel(filepath)
        
        # Create holdings dict
        holdings = {}
        total_value = df['Cur. val'].sum()
        
        for _, row in df.iterrows():
            symbol = row['Instrument']
            weight = row['Cur. val'] / total_value if total_value > 0 else 0
            holdings[symbol] = weight
        
        return holdings, total_value
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        return {}, 0


def plot_portfolio_comparison(current_holdings, recommendations):
    """
    Plot comparison of current portfolio vs. recommended portfolio over last 1 year
    """
    try:
        data_fetcher = DataFetcher()
        
        # Get current portfolio returns
        current_returns = None
        for symbol, weight in current_holdings.items():
            try:
                price_data = data_fetcher.fetch_price_data(symbol, exchange='NSE_EQ')
                if price_data is not None and len(price_data) > 0:
                    returns = price_data.pct_change().dropna()
                    if current_returns is None:
                        current_returns = returns * weight
                    else:
                        current_returns = current_returns.add(returns * weight, fill_value=0)
            except:
                pass
        
        if current_returns is None or len(current_returns) == 0:
            st.warning("⚠️ Could not fetch historical data for portfolio comparison")
            st.write("Debug: current_returns is None or empty")
            return
        
        # Build recommended portfolio (drop + add)
        recommended_holdings = current_holdings.copy()
        
        # Remove dropped assets
        for asset in recommendations.get('assets_to_drop', []):
            recommended_holdings.pop(asset['symbol'], None)
        
        # Add new assets (equal weight distribution for new assets)
        new_assets = recommendations.get('assets_to_add', [])
        if new_assets:
            # Calculate total weight to redistribute
            dropped_weight = sum([current_holdings.get(a['symbol'], 0) for a in recommendations.get('assets_to_drop', [])])
            new_weight_per_asset = dropped_weight / len(new_assets) if new_assets else 0
            
            for asset in new_assets[:3]:  # Limit to top 3 to avoid too many tickers
                recommended_holdings[asset['symbol']] = new_weight_per_asset
        
        # Normalize weights to sum to 1
        total_weight = sum(recommended_holdings.values())
        if total_weight > 0:
            recommended_holdings = {k: v / total_weight for k, v in recommended_holdings.items()}
        
        # Get recommended portfolio returns
        recommended_returns = None
        for symbol, weight in recommended_holdings.items():
            try:
                price_data = data_fetcher.fetch_price_data(symbol, exchange='NSE_EQ')
                if price_data is not None and len(price_data) > 0:
                    returns = price_data.pct_change().dropna()
                    if recommended_returns is None:
                        recommended_returns = returns * weight
                    else:
                        recommended_returns = recommended_returns.add(returns * weight, fill_value=0)
            except:
                pass
        
        if recommended_returns is None or len(recommended_returns) == 0:
            st.warning("⚠️ Could not fetch historical data for recommended portfolio")
            return
        
        # Align dates (use common dates)
        common_dates = current_returns.index.intersection(recommended_returns.index)
        
        if len(common_dates) < 50:
            st.warning("⚠️ Not enough historical data for comparison")
            return
        
        # Get data aligned and sorted by date
        current_returns_aligned = current_returns.loc[common_dates].sort_index()
        recommended_returns_aligned = recommended_returns.loc[common_dates].sort_index()
        
        # Sum across all assets to get portfolio-level returns
        if isinstance(current_returns_aligned, pd.DataFrame):
            current_returns_aligned = current_returns_aligned.sum(axis=1)
        if isinstance(recommended_returns_aligned, pd.DataFrame):
            recommended_returns_aligned = recommended_returns_aligned.sum(axis=1)
        
        # Use all available aligned data (already covers last 1 year from DataFetcher)
        if len(current_returns_aligned) < 50:
            st.warning("⚠️ Not enough historical data for comparison")
            return
        
        # Calculate cumulative returns as decimal (not percentage yet)
        current_cumulative = (1 + current_returns_aligned).cumprod() - 1
        recommended_cumulative = (1 + recommended_returns_aligned).cumprod() - 1
        
        # Remove duplicates, keep last occurrence
        current_cumulative = current_cumulative[~current_cumulative.index.duplicated(keep='last')]
        recommended_cumulative = recommended_cumulative[~recommended_cumulative.index.duplicated(keep='last')]
        
        # Convert to percentage for display
        current_cumulative_pct = current_cumulative * 100
        recommended_cumulative_pct = recommended_cumulative * 100
        
        # Convert index to proper datetime
        current_dates = pd.to_datetime(current_cumulative.index)
        recommended_dates = pd.to_datetime(recommended_cumulative.index)
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=current_dates,
            y=current_cumulative_pct.values,
            name='Current Portfolio',
            line=dict(color='#FF6B6B', width=2),
            fill='tozeroy',
            hovertemplate='<b>Current Portfolio</b><br>Date: %{x|%b %d, %Y}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=recommended_dates,
            y=recommended_cumulative_pct.values,
            name='Recommended Portfolio',
            line=dict(color='#4ECDC4', width=2),
            fill='tozeroy',
            hovertemplate='<b>Recommended Portfolio</b><br>Date: %{x|%b %d, %Y}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Generate monthly date range from Jan 2025 to Jan 2026 for x-axis
        date_range_start = pd.Timestamp('2025-01-01')
        date_range_end = pd.Timestamp('2026-01-31')
        monthly_dates = pd.date_range(start=date_range_start, end=date_range_end, freq='MS')
        
        fig.update_layout(
            title='Portfolio Returns Comparison (Jan 2025 - Jan 2026)',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            template='plotly_dark',
            height=450,
            xaxis=dict(
                tickformat='%b %Y',
                dtick='M1',
                tickvals=monthly_dates,
                range=[date_range_start, date_range_end]
            ),
            yaxis=dict(
                ticksuffix='%',
                tickformat='.2f'
            )
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Calculate and display risk metrics comparison
        st.markdown("---")
        st.subheader("📊 Risk & Return Metrics Comparison")
        
        # Calculate metrics for both portfolios
        from portfolio_metrics import PortfolioMetrics
        metrics_calculator = PortfolioMetrics()
        
        # Current portfolio metrics
        current_var_95 = metrics_calculator.calculate_var(current_returns_aligned)
        current_var_99 = metrics_calculator.calculate_var(current_returns_aligned, confidence_level=0.99)
        current_sharpe = metrics_calculator.calculate_sharpe_ratio(current_returns_aligned)
        current_volatility = current_returns_aligned.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Recommended portfolio metrics
        recommended_var_95 = metrics_calculator.calculate_var(recommended_returns_aligned)
        recommended_var_99 = metrics_calculator.calculate_var(recommended_returns_aligned, confidence_level=0.99)
        recommended_sharpe = metrics_calculator.calculate_sharpe_ratio(recommended_returns_aligned)
        recommended_volatility = recommended_returns_aligned.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "VaR (95%)",
                f"{current_var_95:.1f}%",
                delta=f"{recommended_var_95 - current_var_95:+.1f}%",
                delta_color="normal" if recommended_var_95 < current_var_95 else "inverse"
            )
        
        with col2:
            st.metric(
                "VaR (99%)",
                f"{current_var_99:.1f}%",
                delta=f"{recommended_var_99 - current_var_99:+.1f}%",
                delta_color="normal" if recommended_var_99 < current_var_99 else "inverse"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{current_sharpe:.2f}",
                delta=f"{recommended_sharpe - current_sharpe:+.2f}",
                delta_color="normal" if recommended_sharpe > current_sharpe else "inverse"
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{current_volatility:.1f}%",
                delta=f"{recommended_volatility - current_volatility:+.1f}%",
                delta_color="normal" if recommended_volatility < current_volatility else "inverse"
            )
        
        # Risk-Return Trade-off Analysis
        st.markdown("### 🎯 Risk-Return Trade-off Analysis")
        
        # Create risk-return scatter plot
        risk_return_fig = go.Figure()
        
        risk_return_fig.add_trace(go.Scatter(
            x=[current_volatility],
            y=[current_cumulative_pct.iloc[-1]],
            mode='markers+text',
            marker=dict(size=15, color='#FF6B6B', symbol='star'),
            text=['Current Portfolio'],
            textposition='top center',
            name='Current Portfolio',
            hovertemplate='<b>Current Portfolio</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
        ))
        
        risk_return_fig.add_trace(go.Scatter(
            x=[recommended_volatility],
            y=[recommended_cumulative_pct.iloc[-1]],
            mode='markers+text',
            marker=dict(size=15, color='#4ECDC4', symbol='star'),
            text=['Recommended Portfolio'],
            textposition='top center',
            name='Recommended Portfolio',
            hovertemplate='<b>Recommended Portfolio</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
        ))
        
        risk_return_fig.update_layout(
            title='Risk-Return Profile Comparison',
            xaxis_title='Annualized Volatility (%)',
            yaxis_title='Total Return (%)',
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(risk_return_fig, width='stretch')
        
        # Analysis interpretation
        return_diff = recommended_cumulative_pct.iloc[-1] - current_cumulative_pct.iloc[-1]
        volatility_diff = recommended_volatility - current_volatility
        sharpe_diff = recommended_sharpe - current_sharpe
        
        st.markdown("### 📋 Portfolio Change Analysis")
        
        if return_diff < -10:  # Significant return reduction
            st.warning("⚠️ **Significant Return Reduction Detected**")
            st.write(f"""
            The recommended portfolio shows a **{abs(return_diff):.1f}% reduction in returns** but offers:
            - **{abs(volatility_diff):.1f}% lower volatility** (more stable)
            - **{sharpe_diff:+.2f} Sharpe ratio change** ({'improvement' if sharpe_diff > 0 else 'decline'})
            
            **This appears to be a conservative rebalancing strategy** that prioritizes risk reduction over return maximization.
            Consider your risk tolerance and investment timeline before implementing.
            """)
        elif return_diff > 5:  # Significant return improvement
            st.success("✅ **Return Improvement with Risk Adjustment**")
            st.write(f"""
            The recommended portfolio offers **{return_diff:.1f}% better returns** with:
            - **{volatility_diff:+.1f}% volatility change** ({'increased' if volatility_diff > 0 else 'reduced'} risk)
            - **{sharpe_diff:+.2f} Sharpe ratio change** ({'improvement' if sharpe_diff > 0 else 'decline'})
            
            **This is a positive risk-adjusted improvement.**
            """)
        else:  # Moderate changes
            st.info("ℹ️ **Moderate Portfolio Adjustment**")
            st.write(f"""
            The recommended portfolio shows **{return_diff:+.1f}% return change** and **{volatility_diff:+.1f}% volatility change**.
            Sharpe ratio: **{sharpe_diff:+.2f}**
            
            **Balanced approach** with moderate risk-return adjustments.
            """)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            last_val = current_cumulative.iloc[-1]
            # Extract scalar value - already in decimal form
            current_total_return = float(last_val) if isinstance(last_val, (int, float, np.number)) else float(last_val.values[0]) if len(current_cumulative) > 0 else 0
            st.metric(
                "Current Portfolio Return",
                f"{current_total_return*100:.2f}%",
                delta=None
            )
        
        with col2:
            last_val = recommended_cumulative.iloc[-1]
            # Extract scalar value - already in decimal form
            recommended_total_return = float(last_val) if isinstance(last_val, (int, float, np.number)) else float(last_val.values[0]) if len(recommended_cumulative) > 0 else 0
            st.metric(
                "Recommended Portfolio Return",
                f"{recommended_total_return*100:.2f}%",
                delta=None
            )
        
        with col3:
            improvement = (recommended_total_return - current_total_return) * 100
            st.metric(
                "Potential Improvement",
                f"{improvement:.2f}%",
                delta=f"{improvement:+.2f}%",
                delta_color="normal" if improvement > 0 else "inverse"
            )
    
    except Exception as e:
        st.warning(f"⚠️ Could not create portfolio comparison: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")


def main():
    st.markdown('<p class="main-header">🎯 Asset Recommendations (Nifty50)</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ⚠️ **Important**: You MUST first fill in your Investor Profile on the **Main Dashboard (Port 8501)**.
    
    The Asset Recommendations dashboard uses your investor profile data (IID) that you save there.
    Once saved, this dashboard will use your profile for better recommendations.
    """)
    
    # Initialize session state
    if 'investor_profile_loaded' not in st.session_state:
        st.session_state.investor_profile_loaded = False
    
    # Try to load existing IID data
    iid_data = None
    try:
        iid_path = Path("IID_filled.json")
        if iid_path.exists():
            with open(iid_path, 'r') as f:
                iid_data = json.load(f)
                st.session_state.investor_profile_loaded = True
    except Exception as e:
        pass
    
    if not st.session_state.investor_profile_loaded:
        st.warning("📋 No investor profile found. Please:")
        st.markdown("""
        1. Go to the **Main Dashboard** (http://localhost:8501)
        2. Fill in your Investor Profile in the form
        3. Click "💾 Save & Continue"
        4. Then come back here and refresh this page
        """)
        return
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show loaded profile
        if iid_data:
            profile_info = iid_data.get('investor_profile', {})
            st.success(f"✅ Investor Profile Loaded")
            st.caption(f"Age: {profile_info.get('age_band', 'N/A')} | Tier: {profile_info.get('city_tier', 'N/A')}")
        
        # Portfolio upload or selection
        st.subheader("1️⃣ Load Your Portfolio")
        portfolio_file = st.file_uploader("Upload Portfolio Excel File", type=["xlsx", "xls"])
        
        if portfolio_file:
            holdings, total_value = load_portfolio_from_file(portfolio_file)
            st.success(f"✅ Portfolio loaded: {len(holdings)} holdings, ₹{total_value:,.0f}")
        else:
            st.info("Upload your portfolio Excel file to get started")
            holdings = {}
            total_value = 0
        
        # Investment objective
        st.subheader("2️⃣ Investment Objective")
        objective = st.radio(
            "Select your investment objective:",
            ["Conservative Income", "Moderate Growth", "Aggressive Growth"],
            index=1
        )
        
        # Number of recommendations
        st.subheader("3️⃣ Settings")
        max_add = st.slider("Max assets to recommend adding:", 1, 10, 5)
        max_drop = st.slider("Max assets to recommend dropping:", 1, 10, 3)
        
        analyze_button = st.button("🔍 Analyze & Recommend", type="primary", key="analyze_btn")
    
    # Main content
    if not holdings:
        st.warning("📁 Please upload your portfolio file in the sidebar to get started.")
        return
    
    if not analyze_button:
        st.info("Click '🔍 Analyze & Recommend' in the sidebar to generate recommendations.")
        return
    
    # Run analysis
    with st.spinner("🔄 Analyzing Nifty50 stocks and generating recommendations..."):
        try:
            # Create optimizer (Nifty50 only)
            # Use the loaded IID data if available, otherwise empty dict
            investor_profile = {}
            if iid_data:
                investor_profile = iid_data.get('investor_profile', {})
            
            optimizer = PortfolioOptimizer(
                current_holdings=holdings,
                investor_profile=investor_profile,
                investment_objective=objective
            )
            
            # Limit to Nifty50 only
            optimizer.available_assets = optimizer.NIFTY50
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/3: Fetching portfolio price data...")
            progress_bar.progress(20)
            
            # Get portfolio returns
            portfolio_returns_data = []
            for symbol in holdings.keys():
                try:
                    price_data = optimizer.data_fetcher.fetch_price_data(symbol, exchange='NSE_EQ')
                    if price_data is not None and len(price_data) > 0:
                        returns = price_data.pct_change().dropna()
                        portfolio_returns_data.append(returns)
                except:
                    pass
            
            progress_bar.progress(40)
            status_text.text("Step 2/3: Analyzing underperformers...")
            
            # Combine portfolio returns
            if portfolio_returns_data:
                portfolio_returns = pd.concat(portfolio_returns_data, axis=1).mean(axis=1)
            else:
                portfolio_returns = pd.Series()
            
            progress_bar.progress(60)
            status_text.text("Step 3/3: Scoring Nifty50 assets...")
            
            # Generate recommendations
            recommendations = optimizer.generate_recommendations(
                portfolio_returns=portfolio_returns,
                current_holdings_df=None,
                max_additions=max_add,
                max_removals=max_drop
            )
            
            progress_bar.progress(100)
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())
            return
    
    # Display Results
    st.markdown("---")
    st.header("📊 Recommendations")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Holdings", len(holdings))
    with col2:
        st.metric("Assets to Add", len(recommendations.get('assets_to_add', [])))
    with col3:
        st.metric("Assets to Drop", len(recommendations.get('assets_to_drop', [])))
    
    st.markdown("---")
    
    # Portfolio Comparison Chart
    st.subheader("📈 Portfolio Returns Comparison")
    st.info("Below shows how your portfolio would have performed with the recommended changes over the last year.")
    plot_portfolio_comparison(holdings, recommendations)
    
    st.markdown("---")
    
    # Assets to DROP
    st.subheader("❌ Recommended Assets to DROP")
    assets_to_drop = recommendations.get('assets_to_drop', [])
    
    if assets_to_drop:
        st.warning("These assets are underperforming. Consider replacing them with better alternatives.")
        
        drop_data = []
        for asset in assets_to_drop:
            drop_data.append({
                'Symbol': asset['symbol'],
                'Current Weight': f"{asset['weight']:.1f}%",
                'Asset Return': f"{asset['asset_return']:.1f}%",
                'Underperformance': f"{asset['underperformance']:.2f}",
                'Action': asset['recommendation']
            })
        
        drop_df = pd.DataFrame(drop_data)
        st.dataframe(drop_df, width='content')
    else:
        st.success("✅ All current holdings are performing adequately!")
    
    st.markdown("---")
    
    # Assets to ADD
    st.subheader("✅ Recommended Assets to ADD")
    
    if recommendations.get('assets_to_add'):
        st.info("These Nifty50 assets align well with your investment objective and improve diversification.")
        
        assets_to_add = recommendations.get('assets_to_add', [])
        
        for idx, asset in enumerate(assets_to_add, 1):
            with st.expander(
                f"#{idx} **{asset['symbol']}** - Score: {asset['score']:.0f}/100 | "
                f"Return: {asset['metrics'].get('returns', 0):.1f}% | "
                f"Sharpe: {asset['metrics'].get('sharpe', 0):.2f}"
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Score", f"{asset['score']:.0f}/100")
                    metrics = asset.get('metrics', {})
                    st.metric("Annual Return", f"{metrics.get('returns', 0):.1f}%")
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
                
                with col2:
                    st.metric("Volatility", f"{metrics.get('volatility', 0):.1f}%")
                    st.metric("Sortino Ratio", f"{metrics.get('sortino', 0):.2f}")
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1f}%")
                
                with col3:
                    st.write("**Why Add This Asset:**")
                    for reason in asset.get('rationale', []):
                        st.write(f"• {reason}")
    else:
        st.info("No strong candidates to add at this time.")
    
    st.markdown("---")
    
    # Implementation Strategy
    st.subheader("📋 Implementation Roadmap")
    
    strategy = recommendations.get('implementation_strategy', {})
    
    for phase_key in ['phase_1', 'phase_2', 'phase_3']:
        phase = strategy.get(phase_key, {})
        if phase:
            with st.expander(f"📌 {phase.get('title', '')}"):
                for action in phase.get('actions', []):
                    st.write(f"• {action}")
    
    st.markdown("---")
    
    # Expected Impact
    st.subheader("📈 Expected Impact")
    
    impact = recommendations.get('expected_impact', {})
    
    # Show interpretation first
    interpretation = impact.get('interpretation', 'Analysis pending')
    direction = impact.get('overall_impact_direction', 'Mixed')
    color = "🟢" if direction == "Positive" else "🟡" if direction == "Mixed" else "🔴"
    
    st.write(f"**{color} {interpretation}**")
    st.info(impact.get('interpretation', ''))
    
    # Show detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sharpe_improve = impact.get('sharpe_ratio_improvement', 0)
        st.metric(
            "Sharpe Ratio Improvement",
            f"{sharpe_improve:.2f}",
            delta=f"{sharpe_improve:+.2f}",
            delta_color="inverse" if sharpe_improve > 0 else "normal"
        )
        st.caption("Risk-adjusted return improvement")
    
    with col2:
        vol_change = impact.get('volatility_change', 0)
        st.metric(
            "Volatility Reduction",
            f"{vol_change:.2f}%",
            delta=f"{vol_change:+.2f}%" if vol_change < 0 else f"-{vol_change:.2f}%",
            delta_color="normal" if vol_change > 0 else "inverse"
        )
        st.caption("Lower is better (less risk)")
    
    with col3:
        raw_return = impact.get('raw_return_change', 0)
        st.metric(
            "Raw Return Change",
            f"{raw_return:.2f}%",
            delta=f"{raw_return:+.2f}%",
            delta_color="normal" if raw_return > 0 else "inverse"
        )
        st.caption("Absolute return change")
    
    # Show details expandable
    with st.expander("📊 Detailed Metrics"):
        details = impact.get('details', {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Assets to Add:**")
            st.write(f"- Sharpe Ratio: {details.get('assets_to_add_sharpe', 0)}")
            st.write(f"- Volatility: {details.get('assets_to_add_volatility', 0):.2f}%")
        
        with col2:
            st.write("**Assets to Drop:**")
            st.write(f"- Sharpe Ratio: {details.get('assets_to_drop_sharpe', 0)}")
            st.write(f"- Volatility: {details.get('assets_to_drop_volatility', 0):.2f}%")
    
    st.info(f"**Confidence Level**: {impact.get('confidence', 'Unknown')} - Always validate recommendations with your financial advisor.")
    
    st.markdown("---")
    
    # Summary
    st.subheader("📌 Summary")
    
    if assets_to_drop or recommendations.get('assets_to_add'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Recommended Changes:**")
            if assets_to_drop:
                st.write(f"• Drop {len(assets_to_drop)} underperforming asset(s)")
            if recommendations.get('assets_to_add'):
                st.write(f"• Add {len(recommendations['assets_to_add'])} high-quality asset(s)")
            sharpe_improve = impact.get('sharpe_ratio_improvement', 0)
            if sharpe_improve > 0:
                st.write(f"• Expected Sharpe ratio improvement: +{sharpe_improve:.2f}")
            vol_reduce = impact.get('volatility_change', 0)
            if vol_reduce > 0:
                st.write(f"• Expected volatility reduction: {vol_reduce:.2f}%")
        
        with col2:
            st.markdown("**Next Steps:**")
            st.write("1. Review the recommended assets")
            st.write("2. Validate with your financial advisor")
            st.write("3. Execute in phases (see Implementation Roadmap)")
            st.write("4. Monitor the new holdings for 4 weeks")
    else:
        st.success("✅ Your portfolio is well-optimized. No major changes needed.")


if __name__ == "__main__":
    main()
