import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

term_years = st.sidebar.slider(
    "Agreement Term (years)", 
    min_value=3, 
    max_value=5, 
    value=5, 
    step=1
)

# Trading performance parameters
st.sidebar.subheader("Trading Performance Scenarios")
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    pessimistic_apr = st.number_input("Pessimistic APR (%)", min_value=10.0, max_value=40.0, value=25.0, step=0.5)
with col2:
    balanced_apr = st.number_input("Balanced APR (%)", min_value=20.0, max_value=50.0, value=38.0, step=0.5)
with col3:
    optimistic_apr = st.number_input("Optimistic APR (%)", min_value=30.0, max_value=60.0, value=50.0, step=0.5)

# Financing terms
st.sidebar.subheader("Financing Terms")
annual_hurdle_rate = st.sidebar.slider(
    "Annual Hurdle Rate (%)", 
    min_value=8.0, 
    max_value=15.0, 
    value=12.0, 
    step=0.5
)

# Static values (no longer user-configurable)
premium_threshold = 30.0  # Fixed at 30%
premium_share = 50.0      # Fixed at 50%

# Strategy settings
st.sidebar.subheader("Investment Strategy")
strategy_option = st.sidebar.selectbox(
    "Capitalization Strategy",
    ["Always Capitalize", "Always Withdraw"]
)

def calculate_annual_scenario(initial_capital, strategy_apr, hurdle_rate_pct, premium_threshold_pct, premium_share_pct, term_years, decisions=None):
    """
    Calculate year-by-year scenario following the exact logic from your model
    """
    results = []
    current_capital = initial_capital
    total_hurdle_payments = 0
    total_premium_payments = 0
    total_withdrawn = 0
    
    for year in range(1, term_years + 1):
        # Calculate profit from trading
        profit_from_trading = current_capital * (strategy_apr / 100)
        
        # Calculate premium threshold (30% of current capital)
        premium_threshold_amount = current_capital * (premium_threshold_pct / 100)
        
        # Calculate surplus above 30%
        surplus_above_threshold = max(0, profit_from_trading - premium_threshold_amount)
        
        # Calculate investor premium (50% of surplus)
        investor_premium = surplus_above_threshold * (premium_share_pct / 100)
        
        # Calculate hurdle rate (12% of current capital)
        hurdle_rate_payment = current_capital * (hurdle_rate_pct / 100)
        
        # Total return to investor
        total_return = hurdle_rate_payment + investor_premium
        
        # Determine if capitalizing or withdrawing
        if strategy_option == "Always Capitalize":
            capitalize = True
        elif strategy_option == "Always Withdraw":
            capitalize = False
        else:
            capitalize = True  # Default
        
        if capitalize:
            # Capital post capitalization
            capital_post_capitalization = current_capital + total_return
            withdrawn_this_year = 0
        else:
            # Withdraw the payment
            capital_post_capitalization = current_capital
            withdrawn_this_year = total_return
            total_withdrawn += withdrawn_this_year
        
        # Store results
        results.append({
            'Year': year,
            'Initial_Capital': current_capital,
            'Profit_from_Trading': profit_from_trading,
            'Premium_Threshold': premium_threshold_amount,
            'Surplus_Above_Threshold': surplus_above_threshold,
            'Investor_Premium': investor_premium,
            'Hurdle_Rate_Payment': hurdle_rate_payment,
            'Total_Return': total_return,
            'Decision': "Capitalize" if capitalize else "Withdraw",
            'Withdrawn_This_Year': withdrawn_this_year,
            'Capital_Post_Capitalization': capital_post_capitalization,
            'Strategy_APR': strategy_apr
        })
        
        # Update capital for next year
        current_capital = capital_post_capitalization
        total_hurdle_payments += hurdle_rate_payment
        total_premium_payments += investor_premium
    
    return results, total_hurdle_payments, total_premium_payments, total_withdrawn

# Calculate scenarios
scenarios = {
    'Pessimistic': pessimistic_apr,
    'Balanced': balanced_apr, 
    'Optimistic': optimistic_apr
}

all_results = {}
summary_data = {}

for scenario_name, apr in scenarios.items():
    results, total_hurdle, total_premium, total_withdrawn = calculate_annual_scenario(
        initial_investment, 
        apr, 
        annual_hurdle_rate, 
        premium_threshold, 
        premium_share, 
        term_years
    )
    
    all_results[scenario_name] = results
    
    final_capital = results[-1]['Capital_Post_Capitalization']
    total_return_eur = (final_capital + total_withdrawn) - initial_investment
    total_return_pct = ((final_capital + total_withdrawn) / initial_investment - 1) * 100
    annual_avg_return = ((final_capital + total_withdrawn) / initial_investment) ** (1/term_years) - 1
    
    summary_data[scenario_name] = {
        'Strategy_Performance_APR': f"{apr}%",
        'Final_Capital': final_capital,
        'Total_Withdrawn': total_withdrawn,
        'Total_Value': final_capital + total_withdrawn,
        'Total_Return_EUR': total_return_eur,
        'Total_Return_PCT': total_return_pct,
        'Annual_Average_Return': annual_avg_return * 100,
        'Hurdle_Rate_Component': total_hurdle,
        'Premium_Component': total_premium,
        'Premium_Share_in_Total': (total_premium / total_return_eur * 100) if total_return_eur > 0 else 0
    }

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Scenario Overview", "📊 Year-by-Year Analysis", "💰 Summary Comparison"])

with tab1:
    # Key metrics for all scenarios
    st.subheader("Key Metrics Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    scenarios_list = ['Pessimistic', 'Balanced', 'Optimistic']
    
    for i, scenario in enumerate(scenarios_list):
        data = summary_data[scenario]
        with [col1, col2, col3][i]:
            st.markdown(f"### {scenario}")
            st.metric(
                "Total Value", 
                f"€{data['Total_Value']:,.0f}",
                f"+{data['Total_Return_PCT']:.1f}%"
            )
            st.metric(
                "Annual Avg Return", 
                f"{data['Annual_Average_Return']:.1f}%"
            )
            st.metric(
                "Final Capital", 
                f"€{data['Final_Capital']:,.0f}"
            )
            if data['Total_Withdrawn'] > 0:
                st.metric(
                    "Total Withdrawn", 
                    f"€{data['Total_Withdrawn']:,.0f}"
                )
    
    # Capital growth chart
    st.subheader("Capital Growth Over Time")
    
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green']
    for i, (scenario_name, results) in enumerate(all_results.items()):
        years = [0] + [r['Year'] for r in results]
        capitals = [initial_investment] + [r['Capital_Post_Capitalization'] for r in results]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=capitals,
            mode='lines+markers',
            name=f"{scenario_name} ({scenarios[scenario_name]}% APR)",
            line=dict(color=colors[i], width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Capital Growth Comparison",
        xaxis_title="Year",
        yaxis_title="Capital (€)",
        height=500,
        hovermode='x unified',
        yaxis=dict(tickformat=',.0f')
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Year-by-Year Detailed Analysis")
    
    selected_scenario = st.selectbox("Select Scenario for Detailed View:", scenarios_list)
    
    results_data = all_results[selected_scenario]
    df_detailed = pd.DataFrame(results_data)
    
    # Format the dataframe for display
    df_display = df_detailed.copy()
    currency_columns = ['Initial_Capital', 'Profit_from_Trading', 'Premium_Threshold', 
                       'Surplus_Above_Threshold', 'Investor_Premium', 'Hurdle_Rate_Payment', 
                       'Total_Return', 'Withdrawn_This_Year', 'Capital_Post_Capitalization']
    
    for col in currency_columns:
        df_display[col] = df_display[col].apply(lambda x: f"€{x:,.0f}")
    
    # Rename columns for better display (now with static values)
    df_display = df_display.rename(columns={
        'Initial_Capital': 'Initial Capital',
        'Profit_from_Trading': 'Profit from Trading',
        'Premium_Threshold': 'Premium Threshold (30%)',
        'Surplus_Above_Threshold': 'Surplus Above 30%',
        'Investor_Premium': 'Investor Premium (50%)',
        'Hurdle_Rate_Payment': f'Hurdle Rate ({annual_hurdle_rate}%)',
        'Total_Return': 'Total Return',
        'Withdrawn_This_Year': 'Withdrawn This Year',
        'Capital_Post_Capitalization': 'Capital Post Capitalization'
    })
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Annual returns breakdown chart
    st.subheader(f"Annual Returns Breakdown - {selected_scenario}")
    
    years = df_detailed['Year'].tolist()
    hurdle_payments = df_detailed['Hurdle_Rate_Payment'].tolist()
    premium_payments = df_detailed['Investor_Premium'].tolist()
    
    fig_breakdown = go.Figure()
    
    fig_breakdown.add_trace(go.Bar(
        name=f'Hurdle Rate ({annual_hurdle_rate}%)',
        x=years,
        y=hurdle_payments,
        marker_color='lightblue'
    ))
    
    fig_breakdown.add_trace(go.Bar(
        name='Premium (50%)',
        x=years,
        y=premium_payments,
        marker_color='darkblue'
    ))
    
    fig_breakdown.update_layout(
        title='Annual Returns Breakdown',
        xaxis_title='Year',
        yaxis_title='Return (€)',
        barmode='stack',
        height=400,
        yaxis=dict(tickformat=',.0f')
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

with tab3:
    st.subheader("Summary Comparison Table")
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data).T
    
    # Format for display
    summary_display = summary_df.copy()
    summary_display['Final_Capital'] = summary_display['Final_Capital'].apply(lambda x: f"€{x:,.0f}")
    summary_display['Total_Withdrawn'] = summary_display['Total_Withdrawn'].apply(lambda x: f"€{x:,.0f}")
    summary_display['Total_Value'] = summary_display['Total_Value'].apply(lambda x: f"€{x:,.0f}")
    summary_display['Total_Return_EUR'] = summary_display['Total_Return_EUR'].apply(lambda x: f"€{x:,.0f}")
    summary_display['Total_Return_PCT'] = summary_display['Total_Return_PCT'].apply(lambda x: f"+{x:.1f}%")
    summary_display['Annual_Average_Return'] = summary_display['Annual_Average_Return'].apply(lambda x: f"{x:.1f}%")
    summary_display['Hurdle_Rate_Component'] = summary_display['Hurdle_Rate_Component'].apply(lambda x: f"€{x:,.0f}")
    summary_display['Premium_Component'] = summary_display['Premium_Component'].apply(lambda x: f"€{x:,.0f}")
    summary_display['Premium_Share_in_Total'] = summary_display['Premium_Share_in_Total'].apply(lambda x: f"{x:.0f}%")
    
    # Rename columns
    summary_display = summary_display.rename(columns={
        'Strategy_Performance_APR': 'Strategy Performance',
        'Final_Capital': 'Capital at the End',
        'Total_Withdrawn': 'Total Withdrawn',
        'Total_Value': 'Total Value',
        'Total_Return_EUR': 'Total Return €',
        'Total_Return_PCT': 'Total Return %',
        'Annual_Average_Return': 'Annual Average Return',
        'Hurdle_Rate_Component': 'From Hurdle Rate',
        'Premium_Component': 'From Premium',
        'Premium_Share_in_Total': 'Premium Share in Total Return'
    })
    
    st.dataframe(summary_display, use_container_width=True)
    
    # Return components visualization
    st.subheader("Return Components Analysis")
    
    scenarios_names = list(summary_data.keys())
    hurdle_components = [summary_data[s]['Hurdle_Rate_Component'] for s in scenarios_names]
    premium_components = [summary_data[s]['Premium_Component'] for s in scenarios_names]
    
    fig_components = go.Figure()
    
    fig_components.add_trace(go.Bar(
        name='Hurdle Rate Component',
        x=scenarios_names,
        y=hurdle_components,
        marker_color='lightcoral'
    ))
    
    fig_components.add_trace(go.Bar(
        name='Premium Component',
        x=scenarios_names,
        y=premium_components,
        marker_color='darkred'
    ))
    
    fig_components.update_layout(
        title='Total Return Components by Scenario',
        xaxis_title='Scenario',
        yaxis_title='Return Component (€)',
        barmode='stack',
        height=400,
        yaxis=dict(tickformat=',.0f')
    )
    st.plotly_chart(fig_components, use_container_width=True)



# Footer
st.markdown("---")
st.markdown(f"""
**Model Parameters Summary:**
- Initial Investment: €{initial_investment:,}
- Term: {term_years} years
- Hurdle Rate: {annual_hurdle_rate}% annually
- Premium Threshold: 30% (fixed)
- Premium Share: 50% (fixed)
- Strategy: {strategy_option}
""")

st.markdown("""
**Disclaimer**: This model is for illustrative purposes only. Actual investment returns may vary significantly. 
Past performance does not guarantee future results. Please consult with financial advisors before making investment decisions.
""")
