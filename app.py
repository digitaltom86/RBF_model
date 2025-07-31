import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math

# Page configuration
st.set_page_config(
    page_title="Revenue-Based Financing Model",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("💰 Revenue-Based Financing Agreement Model")
st.markdown("**Interactive model for AdmiEngine BVI Ltd. investment scenarios**")

# Sidebar for parameters
st.sidebar.header("📊 Model Parameters")

# Investment parameters
st.sidebar.subheader("Investment Terms")
initial_investment = st.sidebar.number_input(
    "Initial Investment (€)", 
    min_value=100000, 
    max_value=1000000, 
    value=500000, 
    step=50000,
    format="%d"
)

term_months = st.sidebar.slider(
    "Agreement Term (months)", 
    min_value=36, 
    max_value=60, 
    value=60, 
    step=1
)

# Trading performance parameters
st.sidebar.subheader("Trading Performance")
annual_return = st.sidebar.slider(
    "Expected Annual Return (%)", 
    min_value=0.0, 
    max_value=50.0, 
    value=30.0, 
    step=0.5
)

volatility = st.sidebar.slider(
    "Return Volatility (%)", 
    min_value=0.0, 
    max_value=20.0, 
    value=5.0, 
    step=0.5
)

# Financing terms
st.sidebar.subheader("Financing Terms")
hurdle_rate_quarterly = st.sidebar.slider(
    "Quarterly Hurdle Rate (%)", 
    min_value=2.0, 
    max_value=3.75, 
    value=3.0, 
    step=0.25
)

premium_threshold = st.sidebar.slider(
    "Premium Threshold (%)", 
    min_value=25.0, 
    max_value=35.0, 
    value=30.0, 
    step=0.5
)

premium_share = st.sidebar.slider(
    "Premium Share (%)", 
    min_value=20.0, 
    max_value=80.0, 
    value=50.0, 
    step=5.0
)

# Decision strategy
st.sidebar.subheader("Investment Strategy")
default_strategy = st.sidebar.selectbox(
    "Default Strategy",
    ["Always Capitalize", "Always Withdraw", "Optimize Returns", "Mixed Strategy"]
)

# Advanced options
with st.sidebar.expander("🔧 Advanced Options"):
    seed = st.number_input("Random Seed", min_value=1, max_value=1000, value=42)
    monte_carlo_runs = st.number_input("Monte Carlo Simulations", min_value=100, max_value=1000, value=500)

# Helper functions
def generate_quarterly_returns(annual_return, volatility, quarters, seed=42):
    """Generate quarterly returns with volatility"""
    np.random.seed(seed)
    quarterly_base = (1 + annual_return/100) ** (1/4) - 1
    quarterly_returns = np.random.normal(
        quarterly_base, 
        volatility/100/2, 
        quarters
    )
    return quarterly_returns

def calculate_scenario(params):
    """Calculate investment scenario"""
    quarters = math.ceil(params['term_months'] / 3)
    quarterly_returns = generate_quarterly_returns(
        params['annual_return'], 
        params['volatility'], 
        quarters, 
        params['seed']
    )
    
    results = []
    current_capital_base = params['initial_investment']
    total_payments_received = 0
    total_capitalized = 0
    
    for quarter in range(quarters):
        # Calculate quarterly return
        quarterly_return = quarterly_returns[quarter]
        annual_equivalent = ((1 + quarterly_return) ** 4 - 1) * 100
        
        # Calculate hurdle payment
        hurdle_payment = current_capital_base * (params['hurdle_rate_quarterly'] / 100)
        
        # Calculate premium payment (only at year-end quarters)
        premium_payment = 0
        if (quarter + 1) % 4 == 0:  # Year-end
            if annual_equivalent > params['premium_threshold']:
                excess_return = annual_equivalent - params['premium_threshold']
                premium_payment = current_capital_base * (excess_return / 100) * (params['premium_share'] / 100)
        
        total_payment = hurdle_payment + premium_payment
        
        # Decision making based on strategy
        decision = make_quarterly_decision(
            params['strategy'], 
            quarter, 
            annual_equivalent, 
            params['premium_threshold'],
            total_payment,
            current_capital_base
        )
        
        if decision == "capitalize":
            current_capital_base += total_payment
            total_capitalized += total_payment
            cash_received = 0
        else:
            cash_received = total_payment
            total_payments_received += cash_received
        
        results.append({
            'Quarter': quarter + 1,
            'Quarterly_Return': quarterly_return * 100,
            'Annual_Equivalent': annual_equivalent,
            'Current_Capital_Base': current_capital_base,
            'Hurdle_Payment': hurdle_payment,
            'Premium_Payment': premium_payment,
            'Total_Payment': total_payment,
            'Decision': decision,
            'Cash_Received': cash_received,
            'Cumulative_Cash': total_payments_received,
            'Total_Capitalized': total_capitalized
        })
    
    return pd.DataFrame(results), total_payments_received, total_capitalized, current_capital_base

def make_quarterly_decision(strategy, quarter, annual_return, threshold, payment, capital_base):
    """Make quarterly investment decision based on strategy"""
    if strategy == "Always Capitalize":
        return "capitalize"
    elif strategy == "Always Withdraw":
        return "withdraw"
    elif strategy == "Optimize Returns":
        # Capitalize if returns are above threshold, withdraw if below
        return "capitalize" if annual_return > threshold else "withdraw"
    elif strategy == "Mixed Strategy":
        # Capitalize first 2 years, then withdraw
        return "capitalize" if quarter < 8 else "withdraw"
    else:
        return "withdraw"

# Main calculation
params = {
    'initial_investment': initial_investment,
    'term_months': term_months,
    'annual_return': annual_return,
    'volatility': volatility,
    'hurdle_rate_quarterly': hurdle_rate_quarterly,
    'premium_threshold': premium_threshold,
    'premium_share': premium_share,
    'strategy': default_strategy,
    'seed': seed
}

# Calculate base scenario
df_results, total_cash, total_capitalized, final_capital = calculate_scenario(params)

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📊 Quarterly Analysis", "💰 Cash Flow", "🎯 Scenario Comparison", "🎲 Monte Carlo"])

with tab1:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Cash Received", 
            f"€{total_cash:,.0f}",
            f"{(total_cash / initial_investment - 1) * 100:.1f}% of initial"
        )
    
    with col2:
        st.metric(
            "Total Capitalized", 
            f"€{total_capitalized:,.0f}",
            f"{(total_capitalized / initial_investment) * 100:.1f}% of initial"
        )
    
    with col3:
        st.metric(
            "Final Capital Base", 
            f"€{final_capital:,.0f}",
            f"{(final_capital / initial_investment - 1) * 100:.1f}% growth"
        )
    
    with col4:
        total_value = total_cash + final_capital
        st.metric(
            "Total Value Created", 
            f"€{total_value:,.0f}",
            f"{(total_value / initial_investment - 1) * 100:.1f}% return"
        )
    
    # Summary chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Capital Base Growth", "Quarterly Payments", "Annual Returns", "Cumulative Cash Flow"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Capital base growth
    fig.add_trace(
        go.Scatter(x=df_results['Quarter'], y=df_results['Current_Capital_Base'], 
                  name='Capital Base', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Quarterly payments
    fig.add_trace(
        go.Bar(x=df_results['Quarter'], y=df_results['Total_Payment'], 
               name='Quarterly Payment', marker_color='green'),
        row=1, col=2
    )
    
    # Annual returns
    fig.add_trace(
        go.Scatter(x=df_results['Quarter'], y=df_results['Annual_Equivalent'], 
                  name='Annual Return %', line=dict(color='red')),
        row=2, col=1
    )
    
    # Cumulative cash flow
    fig.add_trace(
        go.Scatter(x=df_results['Quarter'], y=df_results['Cumulative_Cash'], 
                  name='Cumulative Cash', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Investment Overview Dashboard")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Quarterly Decision Analysis")
    
    # Interactive quarterly decisions
    st.markdown("### Override Quarterly Decisions")
    decision_override = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("**Quick Actions:**")
        if st.button("Capitalize All"):
            for i in range(len(df_results)):
                decision_override[i] = "capitalize"
        if st.button("Withdraw All"):
            for i in range(len(df_results)):
                decision_override[i] = "withdraw"
        if st.button("Reset to Strategy"):
            decision_override = {}
    
    with col1:
        # Display quarterly details with decision options
        for idx, row in df_results.iterrows():
            with st.expander(f"Quarter {row['Quarter']} - {row['Decision'].title()} (€{row['Total_Payment']:.0f})"):
                col_a, col_b, col_c = st.columns([1, 1, 1])
                
                with col_a:
                    st.write(f"**Return:** {row['Annual_Equivalent']:.1f}%")
                    st.write(f"**Capital Base:** €{row['Current_Capital_Base']:.0f}")
                
                with col_b:
                    st.write(f"**Hurdle:** €{row['Hurdle_Payment']:.0f}")
                    st.write(f"**Premium:** €{row['Premium_Payment']:.0f}")
                
                with col_c:
                    override_decision = st.selectbox(
                        "Decision Override:",
                        ["Use Strategy", "Capitalize", "Withdraw"],
                        key=f"decision_{idx}"
                    )
                    if override_decision != "Use Strategy":
                        decision_override[idx] = override_decision.lower()
    
    # Recalculate if there are overrides
    if decision_override:
        st.info(f"Recalculating with {len(decision_override)} decision overrides...")
        # This would require a more complex recalculation function

with tab3:
    st.subheader("Cash Flow Analysis")
    
    # Cash flow waterfall chart
    fig_waterfall = go.Figure(go.Waterfall(
        name="Cash Flow",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Initial Investment", "Cash Received", "Capitalized Amount", "Final Position"],
        textposition="outside",
        text=[f"€{initial_investment:,.0f}", f"€{total_cash:,.0f}", 
              f"€{total_capitalized:,.0f}", f"€{final_capital:,.0f}"],
        y=[initial_investment, total_cash, total_capitalized, final_capital],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig_waterfall.update_layout(
        title="Investment Cash Flow Waterfall",
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Detailed cash flow table
    st.subheader("Detailed Quarterly Cash Flow")
    
    # Add some calculated columns for better analysis
    df_display = df_results.copy()
    df_display['ROI_Quarterly'] = (df_display['Total_Payment'] / df_display['Current_Capital_Base'] * 100).round(2)
    df_display['Cumulative_ROI'] = ((df_display['Cumulative_Cash'] + df_display['Total_Capitalized']) / initial_investment * 100).round(1)
    
    st.dataframe(
        df_display[['Quarter', 'Annual_Equivalent', 'Current_Capital_Base', 
                   'Total_Payment', 'Decision', 'Cash_Received', 'Cumulative_Cash', 'ROI_Quarterly', 'Cumulative_ROI']],
        use_container_width=True
    )

with tab4:
    st.subheader("Scenario Comparison")
    
    # Generate multiple scenarios
    scenarios = {
        "Conservative (20% return)": {**params, 'annual_return': 20, 'strategy': 'Always Withdraw'},
        "Base Case (30% return)": {**params, 'annual_return': 30, 'strategy': 'Mixed Strategy'},
        "Optimistic (40% return)": {**params, 'annual_return': 40, 'strategy': 'Always Capitalize'},
        "High Volatility": {**params, 'volatility': 15, 'strategy': 'Optimize Returns'}
    }
    
    scenario_results = {}
    for name, scenario_params in scenarios.items():
        df_scenario, cash, capitalized, final_cap = calculate_scenario(scenario_params)
        scenario_results[name] = {
            'total_cash': cash,
            'total_capitalized': capitalized,
            'final_capital': final_cap,
            'total_value': cash + final_cap,
            'roi': (cash + final_cap) / initial_investment - 1
        }
    
    # Scenario comparison table
    comparison_df = pd.DataFrame(scenario_results).T
    comparison_df['Total Return %'] = (comparison_df['roi'] * 100).round(1)
    
    st.dataframe(
        comparison_df[['total_cash', 'total_capitalized', 'final_capital', 'total_value', 'Total Return %']],
        use_container_width=True,
        column_config={
            'total_cash': st.column_config.NumberColumn('Cash Received', format='€%.0f'),
            'total_capitalized': st.column_config.NumberColumn('Capitalized', format='€%.0f'),
            'final_capital': st.column_config.NumberColumn('Final Capital', format='€%.0f'),
            'total_value': st.column_config.NumberColumn('Total Value', format='€%.0f'),
            'Total Return %': st.column_config.NumberColumn('Return %', format='%.1f%%')
        }
    )

with tab5:
    st.subheader("Monte Carlo Analysis")
    
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running simulations..."):
            monte_carlo_results = []
            
            for i in range(monte_carlo_runs):
                sim_params = params.copy()
                sim_params['seed'] = i + 1
                _, cash, capitalized, final_cap = calculate_scenario(sim_params)
                monte_carlo_results.append({
                    'run': i + 1,
                    'total_cash': cash,
                    'final_capital': final_cap,
                    'total_value': cash + final_cap,
                    'roi': (cash + final_cap) / initial_investment - 1
                })
            
            mc_df = pd.DataFrame(monte_carlo_results)
            
            # Monte Carlo statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average ROI", f"{mc_df['roi'].mean()*100:.1f}%")
                st.metric("Median ROI", f"{mc_df['roi'].median()*100:.1f}%")
            
            with col2:
                st.metric("Best Case ROI", f"{mc_df['roi'].max()*100:.1f}%")
                st.metric("Worst Case ROI", f"{mc_df['roi'].min()*100:.1f}%")
            
            with col3:
                st.metric("Standard Deviation", f"{mc_df['roi'].std()*100:.1f}%")
                percentile_95 = np.percentile(mc_df['roi'], 5)
                st.metric("5th Percentile", f"{percentile_95*100:.1f}%")
            
            # Monte Carlo distribution
            fig_mc = px.histogram(
                mc_df, x='roi', nbins=50,
                title='Distribution of Returns (Monte Carlo)',
                labels={'roi': 'Return on Investment', 'count': 'Frequency'}
            )
            fig_mc.update_xaxis(tickformat='.1%')
            st.plotly_chart(fig_mc, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This model is for illustrative purposes only. Actual investment returns may vary significantly. 
Past performance does not guarantee future results. Please consult with financial advisors before making investment decisions.
""")
