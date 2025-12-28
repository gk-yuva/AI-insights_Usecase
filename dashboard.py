"""
Portfolio Health Dashboard
Interactive Streamlit dashboard for portfolio analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

from portfolio_analyzer import PortfolioAnalyzer


# Page configuration
st.set_page_config(
    page_title="Portfolio Health Dashboard",
    page_icon="📊",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
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
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def run_analysis(portfolio_path, iid_path):
    """Run portfolio analysis and cache results"""
    analyzer = PortfolioAnalyzer(
        portfolio_path=portfolio_path,
        iid_path=iid_path,
        investment_objective="Moderate Growth"
    )
    return analyzer.run_complete_analysis()


def create_quadrant_chart(pqs, ifs):
    """Create two-dimensional quadrant chart"""
    fig = go.Figure()
    
    # Add quadrant backgrounds
    fig.add_shape(type="rect", x0=0, y0=0, x1=65, y1=65,
                  fillcolor="lightcoral", opacity=0.2, line_width=0)
    fig.add_shape(type="rect", x0=65, y0=0, x1=100, y1=65,
                  fillcolor="lightyellow", opacity=0.2, line_width=0)
    fig.add_shape(type="rect", x0=0, y0=65, x1=65, y1=100,
                  fillcolor="lightyellow", opacity=0.2, line_width=0)
    fig.add_shape(type="rect", x0=65, y0=65, x1=100, y1=100,
                  fillcolor="lightgreen", opacity=0.2, line_width=0)
    
    # Add quadrant lines
    fig.add_hline(y=65, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=65, line_dash="dash", line_color="gray", line_width=1)
    
    # Add portfolio point
    fig.add_trace(go.Scatter(
        x=[pqs],
        y=[ifs],
        mode='markers+text',
        marker=dict(size=20, color='darkblue', symbol='star'),
        text=['Your Portfolio'],
        textposition='top center',
        textfont=dict(size=14, color='darkblue'),
        name='Portfolio'
    ))
    
    # Add quadrant labels
    fig.add_annotation(x=32.5, y=32.5, text="Needs<br>Improvement",
                      showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=82.5, y=32.5, text="Good Portfolio<br>Poor Fit",
                      showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=32.5, y=82.5, text="Poor Portfolio<br>Good Fit",
                      showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=82.5, y=82.5, text="Ideal<br>Zone",
                      showarrow=False, font=dict(size=12, color="gray"))
    
    fig.update_layout(
        title="Portfolio Positioning Matrix",
        xaxis_title="Portfolio Quality Score (PQS)",
        yaxis_title="Investor Fit Score (IFS)",
        xaxis=dict(range=[0, 100], dtick=20),
        yaxis=dict(range=[0, 100], dtick=20),
        height=500,
        showlegend=False
    )
    
    return fig


def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for scores"""
    if value >= 80:
        color = "green"
    elif value >= 65:
        color = "lightgreen"
    elif value >= 50:
        color = "yellow"
    elif value >= 35:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 35], 'color': "rgba(255, 0, 0, 0.1)"},
                {'range': [35, 50], 'color': "rgba(255, 165, 0, 0.1)"},
                {'range': [50, 65], 'color': "rgba(255, 255, 0, 0.1)"},
                {'range': [65, 80], 'color': "rgba(144, 238, 144, 0.1)"},
                {'range': [80, 100], 'color': "rgba(0, 128, 0, 0.1)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 65
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_metrics_comparison(metrics):
    """Create bar chart comparing portfolio metrics"""
    metric_names = ['Sharpe Ratio', 'Sortino Ratio', 'Annual Return (%)', 'Volatility (%)']
    values = [
        metrics.get('sharpe_ratio', 0),
        metrics.get('sortino_ratio', 0),
        metrics.get('annual_return', 0) * 100,
        metrics.get('volatility', 0) * 100
    ]
    
    fig = go.Figure(data=[
        go.Bar(x=metric_names, y=values, marker_color='steelblue')
    ])
    
    fig.update_layout(
        title="Portfolio Performance Metrics",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )
    
    return fig


def create_investor_indices_radar(indices):
    """Create radar chart for investor indices"""
    categories = ['Risk Capacity', 'Risk Tolerance', 'Time Horizon', 
                  'Behavioral Discipline']
    values = [
        indices.get('risk_capacity_index', 50),
        indices.get('risk_tolerance_index', 50),
        indices.get('time_horizon_strength', 50),
        100 - indices.get('behavioral_fragility_index', 50)  # Invert fragility
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name='Your Profile',
        line_color='steelblue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title="Investor Profile Indices",
        height=400
    )
    
    return fig


def save_iid_data(data, file_path="IID_filled.json"):
    """Save filled IID data to JSON file in a safe, repo-relative path.

    Ensures parent directories exist so writing on Streamlit Cloud or other
    environments won't raise FileNotFoundError for non-existent folders.
    Returns the absolute path to the saved file.
    """
    from pathlib import Path

    p = Path(file_path).expanduser()
    # If a directory was provided, create it. For plain filenames this is a no-op.
    if p.parent and not p.parent.exists():
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If mkdir fails (permissions) fall back to current working directory
            p = Path.cwd() / p.name

    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return str(p)


def main():
    # Initialize session state
    if 'iid_saved' not in st.session_state:
        st.session_state.iid_saved = False
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    
    # Header
    st.markdown('<p class="main-header">📊 Portfolio Health Analysis Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        portfolio_path = st.text_input(
            "Portfolio Path",
            value=r"f:\Insights\Portfolio Data_Hypothetical.xlsx"
        )
        
        iid_path = st.text_input(
            "IID Path",
            value=r"F:\AI Insights Dashboard\IID_filled.json"
        )
        
        # Show IID status
        if st.session_state.iid_saved:
            st.success("✅ Investor profile saved!")
        
        analyze_button = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
        
        if analyze_button:
            st.session_state.show_analysis = True
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This dashboard provides a comprehensive two-dimensional analysis:
        
        - **Portfolio Quality (PQS)**: How good is the portfolio?
        - **Investor Fit (IFS)**: Is it right for you?
        
        Our unique approach optimizes for **staying invested**, not just returns.
        """)
    
    # Main content - show form or analysis based on state
    if not st.session_state.show_analysis:
        # Show investor profile form
        st.header("📝 Investor Information")
        st.info("👋 Welcome! Please fill in your details below to get started. This information helps us calculate your Investor Fit Score (IFS).")
        
        with st.form("investor_form"):
            st.subheader("Personal Profile")
            col1, col2 = st.columns(2)
            
            with col1:
                age_band = st.selectbox(
                    "Age Band",
                    ["25-30", "30-35", "35-40", "40-50", "50-60", "60+"],
                    index=1
                )
                employment_type = st.selectbox(
                    "Employment Type",
                    ["salaried", "self_employed", "business", "retired"],
                    index=0
                )
            
            with col2:
                city_tier = st.selectbox(
                    "City Tier",
                    ["tier_1", "tier_2", "tier_3"],
                    index=0
                )
                industry = st.text_input("Industry", value="Technology")
            
            st.subheader("Investment Goals")
            col1, col2 = st.columns(2)
            
            with col1:
                primary_goal = st.selectbox(
                    "Primary Goal",
                    ["wealth_creation", "inflation_beating", "passive_income", 
                     "capital_preservation", "early_retirement", "tax_optimization"],
                    index=0
                )
                target_corpus = st.number_input(
                    "Target Corpus (₹)",
                    min_value=100000,
                    value=10000000,
                    step=100000
                )
            
            with col2:
                earliest_year = st.number_input(
                    "Earliest Liquidity Year",
                    min_value=2025,
                    value=2035,
                    step=1
                )
                comfortable_year = st.number_input(
                    "Comfortable Exit Year",
                    min_value=2025,
                    value=2045,
                    step=1
                )
            
            st.subheader("Risk Profile")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                income_stability = st.selectbox(
                    "Income Stability",
                    ["very_stable", "stable", "variable", "volatile"],
                    index=1
                )
                emergency_fund_months = st.number_input(
                    "Emergency Fund (Months)",
                    min_value=0,
                    max_value=24,
                    value=6
                )
            
            with col2:
                drawdown_response = st.selectbox(
                    "Response to Market Crash",
                    ["invest_more", "hold", "reduce", "exit"],
                    index=1
                )
                sleep_loss_drawdown = st.slider(
                    "Sleep Loss Drawdown (%)",
                    min_value=5,
                    max_value=50,
                    value=25,
                    step=5
                )
            
            with col3:
                exited_before = st.checkbox("Exited During Crash Before")
                reentered = st.checkbox("Re-entered After Exit")
            
            st.subheader("Cash Flow & Liabilities")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                monthly_investment = st.number_input(
                    "Monthly Investment (₹)",
                    min_value=0,
                    value=150000,
                    step=10000
                )
                has_liabilities = st.checkbox("Has Liabilities/EMI", value=True)
            
            with col2:
                monthly_emi = st.number_input(
                    "Monthly EMI (₹)",
                    min_value=0,
                    value=35000 if has_liabilities else 0,
                    step=5000,
                    disabled=not has_liabilities
                )
                sip_stepup = st.number_input(
                    "SIP Step-up (%/year)",
                    min_value=0,
                    max_value=50,
                    value=10,
                    step=5
                )
            
            with col3:
                num_dependents = st.number_input(
                    "Number of Dependents",
                    min_value=0,
                    max_value=10,
                    value=2
                )
                dependency_years = st.number_input(
                    "Dependency Duration (Years)",
                    min_value=0,
                    max_value=50,
                    value=15
                )
            
            st.subheader("Behavioral Factors")
            col1, col2 = st.columns(2)
            
            with col1:
                decision_autonomy = st.selectbox(
                    "Decision Making Style",
                    ["rules_based", "guided", "full_control"],
                    index=1
                )
            
            with col2:
                abandonment_triggers = st.multiselect(
                    "Plan Abandonment Triggers",
                    ["market_crash", "income_loss", "family_pressure", "none"],
                    default=["market_crash", "income_loss"]
                )
            
            submitted = st.form_submit_button("💾 Save & Continue", type="primary", use_container_width=True)
            
            if submitted:
                # Build IID JSON
                iid_data = {
                    "schema_version": "1.0",
                    "investor_id": f"user_{datetime.now().strftime('%Y%m%d')}",
                    "investor_profile": {
                        "age_band": age_band,
                        "employment_type": employment_type,
                        "industry": industry,
                        "city_tier": city_tier
                    },
                    "investment_motivation": {
                        "primary_goals": [primary_goal],
                        "goal_priority_rank": {
                            "primary": primary_goal,
                            "secondary": "inflation_beating"
                        }
                    },
                    "time_horizon": {
                        "earliest_liquidity_year": int(earliest_year),
                        "comfortable_exit_year": int(comfortable_year),
                        "horizon_type": "flexible"
                    },
                    "target_corpus": {
                        "amount": int(target_corpus),
                        "confidence_level": "approximate",
                        "inflation_adjusted": True
                    },
                    "risk_profile": {
                        "risk_capacity": {
                            "income_stability": income_stability,
                            "emergency_fund_months": int(emergency_fund_months),
                            "existing_liabilities": {
                                "has_liabilities": has_liabilities,
                                "monthly_emi_amount": int(monthly_emi)
                            },
                            "portfolio_as_percent_of_net_worth": 45
                        },
                        "risk_tolerance": {
                            "drawdown_response": drawdown_response,
                            "sleep_loss_drawdown_percent": int(sleep_loss_drawdown),
                            "loss_vs_missed_gains": "loss_averse"
                        },
                        "risk_behavior_history": {
                            "exited_during_crash_before": exited_before,
                            "reentered_after_exit": reentered
                        }
                    },
                    "cash_flow_and_investment_style": {
                        "investment_mode": "sip",
                        "monthly_investment_amount": int(monthly_investment),
                        "lump_sum_amount": None,
                        "sip_step_up": {
                            "enabled": sip_stepup > 0,
                            "type": "fixed_percent",
                            "value": int(sip_stepup)
                        },
                        "investment_consistency": "mostly_disciplined"
                    },
                    "tax_profile": {
                        "current_tax_bracket_percent": 30,
                        "tax_regime": "old",
                        "expected_future_tax_change": False,
                        "notes": ""
                    },
                    "family_and_dependents": {
                        "number_of_dependents": int(num_dependents),
                        "dependency_duration_years": int(dependency_years),
                        "major_future_liabilities": ["children_education"],
                        "insurance_status": {
                            "term_insurance_adequate": True,
                            "health_insurance_adequate": False
                        }
                    },
                    "behavioral_triggers": {
                        "plan_abandonment_triggers": abandonment_triggers,
                        "decision_autonomy": decision_autonomy
                    },
                    "meta": {
                        "profile_created_at": datetime.now().isoformat(),
                        "last_updated_at": datetime.now().isoformat(),
                        "data_source": "user_declared"
                    }
                }
                
                # Save to file
                saved_path = save_iid_data(iid_data)
                st.session_state.iid_saved = True
                st.success(f"✅ Profile saved successfully!")
                st.info("👉 Now click '🔍 Run Analysis' in the sidebar to see your personalized results!")
    
    else:
        # Show analysis results
        with st.spinner("🔄 Running analysis..."):

            try:
                report = run_analysis(portfolio_path, iid_path)
                
                if not report:
                    st.error("❌ Analysis failed. Please check the file paths.")
                    return
                
                # Extract data
                metrics = report.get('metrics', {})
                pqs_result = report.get('portfolio_quality_score', {})
                ifs_result = report.get('investor_fit_score', {})
                investor_profile = report.get('investor_profile', {})
                psp_result = report.get('plan_survival_probability', {})
                portfolio = report.get('portfolio', [])
                
                # Tab layout
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Overview",
                    "🎯 Two-Dimensional Analysis",
                    "📈 Portfolio Quality",
                    "👤 Investor Fit",
                    "💼 Portfolio Details"
                ])
                
                # TAB 1: Overview
                with tab1:
                    st.header("Portfolio Overview")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Portfolio Value",
                            f"₹{sum([p.get('Cur. val', 0) for p in portfolio]):,.0f}",
                            delta=f"{sum([p.get('P&L', 0) for p in portfolio]):,.0f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Annual Return",
                            f"{metrics.get('annual_return', 0)*100:.2f}%",
                            delta="vs 7.38% benchmark"
                        )
                    
                    with col3:
                        st.metric(
                            "Sharpe Ratio",
                            f"{metrics.get('sharpe_ratio', 0):.2f}",
                            delta="Risk-adjusted"
                        )
                    
                    with col4:
                        health_score = report.get('health_classification', {}).get('overall_score', 0)
                        st.metric(
                            "Health Score",
                            f"{health_score:.1f}/100",
                            delta=report.get('health_classification', {}).get('status', 'Unknown')
                        )
                    
                    st.markdown("---")
                    
                    # Holdings table
                    st.subheader("📋 Holdings")
                    if portfolio:
                        df_portfolio = pd.DataFrame(portfolio)
                        st.dataframe(
                            df_portfolio[['Instrument', 'Qty.', 'LTP', 'Cur. val', 'P&L', 'Net chg.']],
                            use_container_width=True
                        )
                    
                    # Performance metrics chart
                    st.plotly_chart(create_metrics_comparison(metrics), use_container_width=True)
                
                # TAB 2: Two-Dimensional Analysis
                with tab2:
                    st.header("🎯 Two-Dimensional Analysis")
                    
                    # Top scores
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pqs_score = pqs_result.get('pqs_score', 0)
                        st.plotly_chart(
                            create_gauge_chart(pqs_score, "Portfolio Quality Score"),
                            use_container_width=True
                        )
                    
                    with col2:
                        ifs_score = ifs_result.get('ifs_score', 0)
                        st.plotly_chart(
                            create_gauge_chart(ifs_score, "Investor Fit Score"),
                            use_container_width=True
                        )
                    
                    with col3:
                        psp_score = psp_result.get('psp_score', 0)
                        st.plotly_chart(
                            create_gauge_chart(psp_score, "Plan Survival Probability"),
                            use_container_width=True
                        )
                    
                    # Quadrant chart
                    st.plotly_chart(
                        create_quadrant_chart(pqs_score, ifs_score),
                        use_container_width=True
                    )
                    
                    # Positioning message
                    if pqs_score >= 65 and ifs_score >= 65:
                        st.success("✅ **IDEAL ZONE**: Excellent portfolio that is well-suited for you!")
                    elif pqs_score >= 65 and ifs_score < 65:
                        st.warning("⚠️ **GOOD PORTFOLIO, POOR FIT**: Strong performance, but misaligned with your profile")
                    elif pqs_score < 65 and ifs_score >= 65:
                        st.warning("⚠️ **POOR PORTFOLIO, GOOD FIT**: Portfolio matches your profile but has performance issues")
                    else:
                        st.error("❌ **NEEDS IMPROVEMENT**: Both quality and fit need attention")
                    
                    # PSP interpretation
                    st.markdown("---")
                    st.subheader("⭐ Plan Survival Analysis")
                    st.info(psp_result.get('interpretation', ''))
                    st.caption(f"**Risk Level**: {psp_result.get('risk_level', 'Unknown')}")
                
                # TAB 3: Portfolio Quality Score
                with tab3:
                    st.header("📈 Portfolio Quality Score (PQS)")
                    
                    st.metric(
                        "Overall PQS",
                        f"{pqs_score:.1f}/100",
                        delta=pqs_result.get('category', 'Unknown')
                    )
                    
                    st.info(pqs_result.get('interpretation', ''))
                    
                    # Component breakdown
                    st.subheader("Component Scores")
                    
                    components = pqs_result.get('components', {})
                    weights = pqs_result.get('weights', {})
                    
                    for component, score in components.items():
                        weight = weights.get(component, 0)
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            st.write(f"**{component.replace('_', ' ').title()}**")
                        
                        with col2:
                            st.progress(score / 100)
                        
                        with col3:
                            st.write(f"{score:.1f}/100 ({weight}%)")
                    
                    # Risk metrics
                    st.markdown("---")
                    st.subheader("⚠️ Risk Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("VaR (95%)", f"{metrics.get('var_95', 0)*100:.2f}%")
                        st.metric("VaR (99%)", f"{metrics.get('var_99', 0)*100:.2f}%")
                    
                    with col2:
                        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
                        st.metric("Volatility", f"{metrics.get('volatility', 0)*100:.2f}%")
                    
                    with col3:
                        st.metric("Beta", f"{metrics.get('beta', 0):.3f}")
                        st.metric("Jensen's Alpha", f"{metrics.get('jensen_alpha', 0)*100:.2f}%")
                
                # TAB 4: Investor Fit Score
                with tab4:
                    st.header("👤 Investor Fit Score (IFS)")
                    
                    st.metric(
                        "Overall IFS",
                        f"{ifs_score:.1f}/100",
                        delta=ifs_result.get('category', 'Unknown')
                    )
                    
                    st.info(ifs_result.get('interpretation', ''))
                    
                    # Investor indices radar
                    indices = investor_profile.get('indices', {})
                    st.plotly_chart(
                        create_investor_indices_radar(indices),
                        use_container_width=True
                    )
                    
                    # Investor indices details
                    st.subheader("📊 Investor Profile Indices")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Risk Capacity Index", f"{indices.get('risk_capacity_index', 0):.1f}/100")
                        st.caption("Financial ability to bear risk")
                        
                        st.metric("Risk Tolerance Index", f"{indices.get('risk_tolerance_index', 0):.1f}/100")
                        st.caption("Psychological willingness to accept risk")
                    
                    with col2:
                        st.metric("Behavioral Fragility", f"{indices.get('behavioral_fragility_index', 0):.1f}/100")
                        st.caption("Likelihood of abandoning plan (lower is better)")
                        
                        st.metric("Time Horizon Strength", f"{indices.get('time_horizon_strength', 0):.1f}/100")
                        st.caption("Ability to stay invested long-term")
                    
                    # Mismatches
                    st.markdown("---")
                    st.subheader("⚠️ Identified Mismatches")
                    
                    mismatches = ifs_result.get('mismatches', [])
                    if mismatches:
                        for mismatch in mismatches:
                            severity = mismatch.get('severity', 0)
                            if severity > 30:
                                st.error(f"❌ {mismatch.get('message', '')}")
                            elif severity > 15:
                                st.warning(f"⚠️ {mismatch.get('message', '')}")
                            else:
                                st.info(f"ℹ️ {mismatch.get('message', '')}")
                    else:
                        st.success("✅ No significant mismatches detected")
                    
                    # Diagnostics
                    st.markdown("---")
                    st.subheader("📋 Fit Diagnostics")
                    
                    for diagnostic in ifs_result.get('diagnostics', []):
                        if "❌" in diagnostic:
                            st.error(diagnostic)
                        elif "⚠️" in diagnostic:
                            st.warning(diagnostic)
                        else:
                            st.success(diagnostic)
                
                # TAB 5: Portfolio Details
                with tab5:
                    st.header("💼 Complete Portfolio Details")
                    
                    if portfolio:
                        df_full = pd.DataFrame(portfolio)
                        st.dataframe(df_full, use_container_width=True)
                        
                        # Download button
                        csv = df_full.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Portfolio Data",
                            data=csv,
                            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    # Complete report
                    st.markdown("---")
                    st.subheader("📄 Complete Analysis Report")
                    
                    with st.expander("View JSON Report"):
                        st.json(report)
                
                # Success message
                st.sidebar.success("✅ Analysis completed successfully!")
                st.sidebar.caption(f"Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
