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
st.markdown("**Interactive quarterly model for AdmiEngine BVI Ltd. investment scenarios**")

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

# Strategy settings
st.sidebar.subheader("Investment Strategy")
strategy_option = st.sidebar.selectbox(
    "Capitalization Strategy",
    ["Always Capitalize", "Always Withdraw", "Custom Quarterly Decisions"]
)

# Quarterly decision interface
quarterly_decisions = {}
total_quarters = term_years * 4

if strategy_option == "Custom Quarterly Decisions":
    st.sidebar.write("**Quarterly Decisions:**")
    
    # Group by years for better organization
    for year in range(1, term_years + 1):
        with st.sidebar.expander(f"Year {year} Decisions"):
            for quarter_in_year in range(1, 5):
                quarter_num = (year - 1) * 4 + quarter_in_year
                if quarter_num <= total_quarters:
                    quarterly_decisions[f"Q{quarter_num}"] = st.selectbox(
                        f"Q{quarter_in_year} (Quarter {quarter_num})",
                        ["Capitalize", "Withdraw"],
                        key=f"quarter_{quarter_num}"
                    )
else:
    # Generate default decisions based on strategy
    for quarter in range(1, total_quarters + 1):
        if strategy_option == "Always Capitalize":
            quarterly_decisions[f"Q{quarter}"] = "Capitalize"
        else:  # Always Withdraw
            quarterly_decisions[f"Q{quarter}"] = "Withdraw"

def calculate_quarterly_scenario(initial_capital, strategy_apr, hurdle_rate_pct, premium_threshold_pct, premium_share_pct, term_years, quarterly_decisions):
    """
    Calculate quarterly scenario with quarterly decision-making capability
    """
    results = []
    current_capital = initial_capital
    total_hurdle_payments = 0
    total_premium_payments = 0
    total_withdrawn = 0
    
    total_quarters = term_years * 4
    quarterly_hurdle_rate = hurdle_rate_pct / 4  # Convert annual to quarterly
    
    # Track annual performance for premium calculations
    year_start_capitals = {}  # Track capital at start of each year
    quarterly_profits_by_year = {}  # Track quarterly profits by year
    
    for quarter in range(1, total_quarters + 1):
        year = math.ceil(quarter / 4)
        quarter_in_year = ((quarter - 1) % 4) + 1
        
        # Track year start capital
        if quarter_in_year == 1:
            year_start_capitals[year] = current_capital
            quarterly_profits_by_year[year] = []
        
        # Calculate quarterly profit from trading
        quarterly_profit = current_capital * (strategy_apr / 100 / 4)  # Quarterly profit
        quarterly_profits_by_year[year].append(quarterly_profit)
        
        # Calculate quarterly hurdle rate payment
        hurdle_rate_payment = current_capital * (quarterly_hurdle_rate / 100)
        
        # Calculate premium payment (only at year-end, i.e., Q4 of each year)
        investor_premium = 0
        annual_profit = 0
        premium_threshold_amount = 0
        surplus_above_threshold = 0
        
        if quarter_in_year == 4:  # End of year - calculate premium
            # Calculate annual profit and premium
            annual_profit = sum(quarterly_profits_by_year[year])
            year_start_capital = year_start_capitals[year]
            
            # Premium threshold (30% of year start capital)
            premium_threshold_amount = year_start_capital * (premium_threshold_pct / 100)
            
            # Calculate surplus above threshold
            surplus_above_threshold = max(0, annual_profit - premium_threshold_amount)
            
            # Calculate investor premium (50% of surplus)
            investor_premium = surplus_above_threshold * (premium_share_pct / 100)
        
        # Total quarterly return to investor
        total_quarterly_return = hurdle_rate_payment + investor_premium
        
        # Get quarterly decision
        decision_key = f"Q{quarter}"
        decision = quarterly_decisions.get(decision_key, "Capitalize")
        
        if decision == "Capitalize":
            # Capital post capitalization
            capital_post_capitalization = current_capital + total_quarterly_return
            withdrawn_this_quarter = 0
        else:
            # Withdraw the payment
            capital_post_capitalization = current_capital
            withdrawn_this_quarter = total_quarterly_return
            total_withdrawn += withdrawn_this_quarter
        
        # Store results
        results.append({
            'Quarter': quarter,
            'Year': year,
            'Quarter_in_Year': quarter_in_year,
            'Initial_Capital': current_capital,
            'Quarterly_Profit': quarterly_profit,
            'Annual_Profit': annual_profit if quarter_in_year == 4 else 0,
            'Premium_Threshold': premium_threshold_amount,
            'Surplus_Above_Threshold': surplus_above_threshold,
            'Investor_Premium': investor_premium,
            'Hurdle_Rate_Payment': hurdle_rate_payment,
            'Total_Quarterly_Return': total_quarterly_return,
            'Decision': decision,
            'Withdrawn_This_Quarter': withdrawn_this_quarter,
            'Capital_Post_Capitalization': capital_post_capitalization,
            'Strategy_APR': strategy_apr
        })
        
        # Update capital for next quarter
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
    results, total_hurdle, total_premium, total_withdrawn = calculate_quarterly_scenario(
        initial_investment, 
        apr, 
        annual_hurdle_rate, 
        premium_threshold, 
        premium_share, 
        term_years,
        quarterly_decisions
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
tab1, tab2, tab3, tab4 = st.tabs(["📈 Scenario Overview", "📊 Quarterly Analysis", "📅 Annual Summary", "🎯 Quarterly Decision Builder"])

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
    
    # Capital growth chart (show quarterly progression)
    st.subheader("Capital Growth Over Time (Quarterly)")
    
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green']
    for i, (scenario_name, results) in enumerate(all_results.items()):
        quarters = [0] + [r['Quarter'] for r in results]
        capitals = [initial_investment] + [r['Capital_Post_Capitalization'] for r in results]
        
        fig.add_trace(go.Scatter(
            x=quarters,
            y=capitals,
            mode='lines+markers',
            name=f"{scenario_name} ({scenarios[scenario_name]}% APR)",
            line=dict(color=colors[i], width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Capital Growth Comparison (Quarterly)",
        xaxis_title="Quarter",
        yaxis_title="Capital (€)",
        height=500,
        hovermode='x unified',
        yaxis=dict(tickformat=',.0f')
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Quarterly Decision Analysis")
    
    selected_scenario = st.selectbox("Select Scenario for Detailed View:", scenarios_list)
    
    results_data = all_results[selected_scenario]
    df_detailed = pd.DataFrame(results_data)
    
    # Show quarterly decisions and their impact
    st.markdown("### Quarterly Payments & Decisions")
    
    # Format the dataframe for display
    df_display = df_detailed.copy()
    df_display['Quarter_Label'] = df_display.apply(lambda x: f"Q{x['Quarter_in_Year']} Y{x['Year']}", axis=1)
    
    currency_columns = ['Initial_Capital', 'Hurdle_Rate_Payment', 'Investor_Premium', 
                       'Total_Quarterly_Return', 'Withdrawn_This_Quarter', 'Capital_Post_Capitalization']
    
    for col in currency_columns:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"€{x:,.0f}")
    
    # Select key columns for display
    display_columns = ['Quarter_Label', 'Initial_Capital', 'Hurdle_Rate_Payment', 
                      'Investor_Premium', 'Total_Quarterly_Return', 'Decision', 
                      'Withdrawn_This_Quarter', 'Capital_Post_Capitalization']
    
    # Rename columns for better display
    column_rename = {
        'Quarter_Label': 'Quarter',
        'Initial_Capital': 'Capital Start',
        'Hurdle_Rate_Payment': 'Hurdle Payment',
        'Investor_Premium': 'Premium Payment',
        'Total_Quarterly_Return': 'Total Payment',
        'Withdrawn_This_Quarter': 'Withdrawn',
        'Capital_Post_Capitalization': 'Capital End'
    }
    
    df_display_filtered = df_display[display_columns].rename(columns=column_rename)
    st.dataframe(df_display_filtered, use_container_width=True, hide_index=True)
    
    # Quarterly returns chart
    st.subheader(f"Quarterly Payments - {selected_scenario}")
    
    quarters = df_detailed['Quarter'].tolist()
    hurdle_payments = df_detailed['Hurdle_Rate_Payment'].tolist()
    premium_payments = df_detailed['Investor_Premium'].tolist()
    
    fig_quarterly = go.Figure()
    
    fig_quarterly.add_trace(go.Bar(
        name='Hurdle Rate',
        x=quarters,
        y=hurdle_payments,
        marker_color='lightblue'
    ))
    
    fig_quarterly.add_trace(go.Bar(
        name='Premium',
        x=quarters,
        y=premium_payments,
        marker_color='darkblue'
    ))
    
    fig_quarterly.update_layout(
        title='Quarterly Payments Breakdown',
        xaxis_title='Quarter',
        yaxis_title='Payment (€)',
        barmode='stack',
        height=400,
        yaxis=dict(tickformat=',.0f')
    )
    
    st.plotly_chart(fig_quarterly, use_container_width=True)

with tab3:
    st.subheader("Annual Summary & Comparison")
    
    # Aggregate quarterly results to annual for comparison
    annual_summaries = {}
    
    for scenario_name, results in all_results.items():
        df_results = pd.DataFrame(results)
        
        # Group by year and aggregate
        annual_data = []
        for year in range(1, term_years + 1):
            year_data = df_results[df_results['Year'] == year]
            
            if len(year_data) > 0:
                initial_capital_year = year_data.iloc[0]['Initial_Capital']
                final_capital_year = year_data.iloc[-1]['Capital_Post_Capitalization']
                total_hurdle_year = year_data['Hurdle_Rate_Payment'].sum()
                total_premium_year = year_data['Investor_Premium'].sum()
                total_withdrawn_year = year_data['Withdrawn_This_Quarter'].sum()
                
                annual_data.append({
                    'Year': year,
                    'Initial_Capital': initial_capital_year,
                    'Final_Capital': final_capital_year,
                    'Annual_Hurdle': total_hurdle_year,
                    'Annual_Premium': total_premium_year,
                    'Annual_Total': total_hurdle_year + total_premium_year,
                    'Annual_Withdrawn': total_withdrawn_year
                })
        
        annual_summaries[scenario_name] = annual_data
    
    # Display annual summaries
    selected_scenario_annual = st.selectbox("Select Scenario for Annual Summary:", scenarios_list, key="annual_select")
    
    if selected_scenario_annual in annual_summaries:
        annual_df = pd.DataFrame(annual_summaries[selected_scenario_annual])
        
        # Format for display
        display_df = annual_df.copy()
        currency_cols = ['Initial_Capital', 'Final_Capital', 'Annual_Hurdle', 'Annual_Premium', 'Annual_Total', 'Annual_Withdrawn']
        for col in currency_cols:
            display_df[col] = display_df[col].apply(lambda x: f"€{x:,.0f}")
        
        display_df = display_df.rename(columns={
            'Initial_Capital': 'Initial Capital',
            'Final_Capital': 'Final Capital',
            'Annual_Hurdle': 'Hurdle Payments',
            'Annual_Premium': 'Premium Payments',
            'Annual_Total': 'Total Payments',
            'Annual_Withdrawn': 'Total Withdrawn'
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Final summary comparison
    st.subheader("Final Results Comparison")
    
    comparison_data = []
    for scenario_name, data in summary_data.items():
        comparison_data.append({
            'Scenario': scenario_name,
            'APR': f"{scenarios[scenario_name]}%",
            'Final Capital': f"€{data['Final_Capital']:,.0f}",
            'Total Withdrawn': f"€{data['Total_Withdrawn']:,.0f}",
            'Total Value': f"€{data['Total_Value']:,.0f}",
            'Total Return': f"€{data['Total_Return_EUR']:,.0f}",
            'Return %': f"+{data['Total_Return_PCT']:.1f}%",
            'Annual Avg': f"{data['Annual_Average_Return']:.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Quarterly Decision Builder")
    
    if strategy_option == "Custom Quarterly Decisions":
        st.success("🎯 Custom quarterly decisions are active!")
        
        # Show decision summary
        st.markdown("### Your Quarterly Decision Timeline")
        
        decision_summary = []
        
        for quarter in range(1, total_quarters + 1):
            year = math.ceil(quarter / 4)
            quarter_in_year = ((quarter - 1) % 4) + 1
            decision = quarterly_decisions.get(f"Q{quarter}", "Capitalize")
            
            decision_summary.append({
                'Quarter': f"Q{quarter_in_year}",
                'Year': year,
                'Full_Quarter': f"Q{quarter}",
                'Decision': decision,
                'Action': "💰 Withdraw" if decision == "Withdraw" else "📈 Capitalize"
            })
        
        decision_df = pd.DataFrame(decision_summary)
        
        # Group by year for better visualization
        for year in range(1, term_years + 1):
            year_decisions = decision_df[decision_df['Year'] == year]
            
            st.markdown(f"**Year {year}:**")
            cols = st.columns(4)
            
            for idx, (_, row) in enumerate(year_decisions.iterrows()):
                with cols[idx]:
                    color = "🟢" if row['Decision'] == "Capitalize" else "🔴"
                    st.write(f"{color} {row['Quarter']}: {row['Decision']}")
        
        # Impact analysis
        st.markdown("### Impact Analysis")
        
        # Compare with standard strategies
        comparison_results = {}
        
        strategies_to_compare = {
            "Your Custom Strategy": quarterly_decisions,
            "Always Capitalize": {f"Q{q}": "Capitalize" for q in range(1, total_quarters + 1)},
            "Always Withdraw": {f"Q{q}": "Withdraw" for q in range(1, total_quarters + 1)}
        }
        
        for strategy_name, decisions in strategies_to_compare.items():
            strategy_results = {}
            
            for scenario_name, apr in scenarios.items():
                results, total_hurdle, total_premium, total_withdrawn = calculate_quarterly_scenario(
                    initial_investment, apr, annual_hurdle_rate, premium_threshold, 
                    premium_share, term_years, decisions
                )
                
                final_capital = results[-1]['Capital_Post_Capitalization']
                total_value = final_capital + total_withdrawn
                
                strategy_results[scenario_name] = {
                    'Final_Capital': final_capital,
                    'Total_Withdrawn': total_withdrawn,
                    'Total_Value': total_value,
                    'Return_PCT': ((total_value / initial_investment) - 1) * 100
                }
            
            comparison_results[strategy_name] = strategy_results
        
        # Display comparison
        st.markdown("#### Strategy Performance Comparison")
        
        for scenario_name in scenarios_list:
            st.markdown(f"**{scenario_name} Scenario ({scenarios[scenario_name]}% APR):**")
            
            cols = st.columns(len(strategies_to_compare))
            
            for idx, (strategy_name, results) in enumerate(comparison_results.items()):
                with cols[idx]:
                    result = results[scenario_name]
                    st.metric(
                        f"{strategy_name}",
                        f"€{result['Total_Value']:,.0f}",
                        f"{result['Return_PCT']:+.1f}%"
                    )
        
        # Quarterly decision impact chart
        st.markdown("### Quarterly Decision Impact")
        
        selected_scenario_impact = st.selectbox("Select Scenario for Impact Analysis:", scenarios_list, key="impact_select")
        
        # Get results for selected scenario
        custom_results = all_results[selected_scenario_impact]
        df_custom = pd.DataFrame(custom_results)
        
        # Create impact visualization
        fig_impact = go.Figure()
        
        # Show capital growth with decision points
        fig_impact.add_trace(go.Scatter(
            x=df_custom['Quarter'],
            y=df_custom['Capital_Post_Capitalization'],
            mode='lines+markers',
            name='Capital Growth',
            line=dict(color='blue', width=2),
            marker=dict(
                size=8,
                color=['green' if d == 'Capitalize' else 'red' for d in df_custom['Decision']],
                symbol=['circle' if d == 'Capitalize' else 'diamond' for d in df_custom['Decision']]
            ),
            hovertemplate='<b>Quarter %{x}</b><br>' +
                         'Capital: €%{y:,.0f}<br>' +
                         'Decision: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=df_custom['Decision']
        ))
        
        fig_impact.update_layout(
            title=f'Capital Growth with Quarterly Decisions - {selected_scenario_impact}',
            xaxis_title='Quarter',
            yaxis_title='Capital (€)',
            height=500,
            yaxis=dict(tickformat=',.0f'),
            showlegend=True
        )
        
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # Decision statistics
        capitalize_count = sum(1 for d in quarterly_decisions.values() if d == "Capitalize")
        withdraw_count = sum(1 for d in quarterly_decisions.values() if d == "Withdraw")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Capitalize Decisions", f"{capitalize_count}/{total_quarters}")
        with col2:
            st.metric("Withdraw Decisions", f"{withdraw_count}/{total_quarters}")
        with col3:
            st.metric("Capitalization Rate", f"{(capitalize_count/total_quarters)*100:.1f}%")
        
    else:
        st.info("💡 Select 'Custom Quarterly Decisions' in the sidebar to build your custom quarterly strategy.")
        
        st.markdown("### Standard Strategy Comparison")
        
        # Show comparison between Always Capitalize vs Always Withdraw
        strategies_to_compare = ["Always Capitalize", "Always Withdraw"]
        strategy_comparison = {}
        
        for strategy in strategies_to_compare:
            temp_decisions = {}
            
            for quarter in range(1, total_quarters + 1):
                temp_decisions[f"Q{quarter}"] = "Capitalize" if strategy == "Always Capitalize" else "Withdraw"
            
            temp_results = {}
            for scenario_name, apr in scenarios.items():
                results, total_hurdle, total_premium, total_withdrawn = calculate_quarterly_scenario(
                    initial_investment, apr, annual_hurdle_rate, premium_threshold, 
                    premium_share, term_years, temp_decisions
                )
                
                final_capital = results[-1]['Capital_Post_Capitalization']
                total_value = final_capital + total_withdrawn
                temp_results[scenario_name] = total_value
            
            strategy_comparison[strategy] = temp_results
        
        # Display strategy comparison
        comparison_df = pd.DataFrame(strategy_comparison)
        comparison_df.index.name = "Scenario"
        comparison_df = comparison_df.applymap(lambda x: f"€{x:,.0f}")
        
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("""
        **💡 Tips for Quarterly Decision Making:**
        - **Capitalize early** to benefit from compound growth
        - **Withdraw later** to secure profits when capital base is larger
        - **Mix strategies** based on market conditions and personal needs
        - **Monitor premium payments** - they only occur at year-end (Q4)
        """)

# Footer
st.markdown("---")
st.markdown(f"""
**Model Parameters Summary:**
- Initial Investment: €{initial_investment:,}
- Term: {term_years} years ({term_years * 4} quarters)
- Hurdle Rate: {annual_hurdle_rate}% annually ({annual_hurdle_rate/4}% quarterly)
- Premium Threshold: {premium_threshold}%
- Premium Share: {premium_share}%
- Strategy: {strategy_option}
""")

st.markdown("""
**Key Features:**
- ✅ Quarterly decision making (as per agreement)
- ✅ Premium calculations at year-end only (Q4)
- ✅ Compound growth through capitalization
- ✅ Real-time impact analysis
- ✅ Custom quarterly strategies

**Disclaimer**: This model is for illustrative purposes only. Actual investment returns may vary significantly. 
Please consult with financial advisors before making investment decisions.
""")
