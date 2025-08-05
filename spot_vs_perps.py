import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from utils import (
    STRATEGY_CONFIG, CHART_COLORS, HOURS_PER_YEAR,
    discover_token_pair_markets, get_token_overall_apy,
    load_token_config, load_url_config,
    fetch_staking_data, fetch_rates_data, fetch_jupiter_perps_data, fetch_drift_perps_data,
    plot_staking_apy_chart, plot_lending_borrowing_rates_chart,
    plot_jupiter_perps_borrow_rates_chart, fetch_strategy_data,
    calculate_net_apy, plot_strategy_chart
)

def calculate_sol_fees(jupiter_perps_data: Dict, principal_amount: float, leverage: float, time_period_hours: int) -> Optional[pd.DataFrame]:
    """
    Calculate fees for leveraged SOL trading on Jupiter Perpetuals

    Args:
        jupiter_perps_data (Dict): API response data for Jupiter Perps SOL
        principal_amount (float): The principal amount in USD
        leverage (float): The leverage multiplier
        time_period_hours (int): Number of hours of data to include

    Returns:
        pd.DataFrame: DataFrame with fee calculations over time
    """
    if not jupiter_perps_data or not jupiter_perps_data.get('success'):
        st.error("No valid Jupiter Perps data to calculate fees")
        return None

    records = jupiter_perps_data.get('data', {}).get('records', [])

    if not records:
        st.error("No records found in Jupiter Perps data")
        return None

    # Convert records to DataFrame
    df = pd.DataFrame(records)

    # Convert hourBucket to datetime and sort by time
    df['hourBucket'] = pd.to_datetime(df['hourBucket'])
    df = df.sort_values('hourBucket')

    # Filter data to match the exact time period requested
    from utils import filter_dataframe_by_time_window
    df = filter_dataframe_by_time_window(df, time_period_hours, 'hourBucket')

    if df.empty:
        return None

    # Calculate position size
    position_size = principal_amount * leverage

    # Calculate hourly costs based on position size
    df['hourly_borrow_rate_percent'] = df['avgHourlyBorrowRate']
    df['hourly_borrow_cost'] = (df['hourly_borrow_rate_percent'] / 100) * position_size
    df['daily_borrow_cost'] = df['hourly_borrow_cost'] * 24

    # Add metadata
    df['principal_amount'] = principal_amount
    df['leverage'] = leverage
    df['position_size'] = position_size

    return df

def calculate_drift_fees(drift_perps_data: Dict, principal_amount: float, leverage: float, time_period_hours: int) -> Optional[pd.DataFrame]:
    """
    Calculate fees for leveraged SOL trading on Drift Perpetuals

    Args:
        drift_perps_data (Dict): API response data for Drift Perps
        principal_amount (float): The principal amount in USD
        leverage (float): The leverage multiplier
        time_period_hours (int): Number of hours of data to include

    Returns:
        pd.DataFrame: DataFrame with fee calculations over time
    """
    if not drift_perps_data or drift_perps_data.get('status') != 'ok':
        st.error("No valid Drift Perps data to calculate fees")
        return None

    funding_rates = drift_perps_data.get('fundingRates', [])

    if not funding_rates:
        st.error("No funding rate records found in Drift Perps data")
        return None

    # Convert records to DataFrame
    df = pd.DataFrame(funding_rates)

    # Convert timestamp to datetime and sort by time
    df['hourBucket'] = pd.to_datetime(df['ts'], unit='s')
    df = df.sort_values('hourBucket')

    # Filter data to match the exact time period requested
    if len(df) > 0:
        # Calculate the time cutoff based on the requested time period
        latest_time = df['hourBucket'].max()
        time_cutoff = latest_time - pd.Timedelta(hours=time_period_hours)

        # Filter to only include data within the requested time period
        df = df[df['hourBucket'] >= time_cutoff]

        # If we still have too many records, take the most recent ones
        if len(df) > time_period_hours * 2:  # Allow some buffer for irregular intervals
            df = df.tail(time_period_hours * 2)

    # Convert string fields to numeric types
    df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    df['oraclePriceTwap'] = pd.to_numeric(df['oraclePriceTwap'], errors='coerce')

    # Calculate position size
    position_size = principal_amount * leverage

    # Calculate funding rate hourly percentage
    # Formula: fundingRateHourlyPct = (fundingRate / 1e9) / (oraclePriceTwap / 1e6) * 100
    df['hourly_funding_rate_percent'] = (df['fundingRate'] / 1e9) / (df['oraclePriceTwap'] / 1e6) * 100

    # Calculate hourly costs based on position size
    df['hourly_funding_cost'] = (df['hourly_funding_rate_percent'] / 100) * position_size
    df['daily_funding_cost'] = df['hourly_funding_cost'] * 24

    # Add metadata
    df['principal_amount'] = principal_amount
    df['leverage'] = leverage
    df['position_size'] = position_size

    return df

def fetch_token_pair_market_data(market_config: Dict, limit: int = 168) -> Optional[Dict]:
    """
    Fetch complete data for one token pair market

    Args:
        market_config (Dict): Market configuration from discover_token_pair_markets
        limit (int): Number of hours of data to fetch

    Returns:
        Dict: Complete market data including rates and staking
    """
    try:
        protocol = market_config['protocol']
        token_a_info = market_config['token_a_info']
        token_b_info = market_config['token_b_info']

        # Fetch staking data for both tokens if available
        token_a_staking = None
        if token_a_info.get('hasStakingYield', False):
            token_a_staking = fetch_staking_data(token_a_info['mint'], limit)

        token_b_staking = None
        if token_b_info.get('hasStakingYield', False):
            token_b_staking = fetch_staking_data(token_b_info['mint'], limit)

        # Fetch rates data for both tokens
        token_a_rates = fetch_rates_data(market_config['token_a_bank'], protocol, limit)
        token_b_rates = fetch_rates_data(market_config['token_b_bank'], protocol, limit)

        if not token_a_rates or not token_b_rates:
            return None

        return {
            'rates_data': {
                'token_a_rates': token_a_rates,
                'token_b_rates': token_b_rates
            },
            'staking_data': {
                'token_a_staking': token_a_staking,
                'token_b_staking': token_b_staking
            },
            'metadata': {
                'protocol': protocol,
                'market': market_config['market'],
                'token_a_symbol': STRATEGY_CONFIG['asgard_spot']['token_a'],
                'token_b_symbol': STRATEGY_CONFIG['asgard_spot']['token_b'],
                'token_a_has_staking': token_a_info.get('hasStakingYield', False),
                'token_b_has_staking': token_b_info.get('hasStakingYield', False),
                'market_key': market_config['market_key']
            }
        }

    except Exception as e:
        st.error(f"Error fetching market data for {market_config.get('protocol', 'unknown')}: {str(e)}")
        return None

def fetch_all_asgard_markets_data(principal_amount: float, leverage: float, limit: int = 168) -> Dict:
    """
    Fetch data for all available Asgard markets

    Args:
        principal_amount (float): Principal amount in USD
        leverage (float): Leverage multiplier
        limit (int): Number of hours of data to fetch

    Returns:
        Dict: Data for all available markets
    """
    try:
        token_config = load_token_config()
        token_a = STRATEGY_CONFIG['asgard_spot']['token_a']
        token_b = STRATEGY_CONFIG['asgard_spot']['token_b']

        # Discover all available markets
        available_markets = discover_token_pair_markets(token_config, token_a, token_b)

        if not available_markets:
            st.error(f"No markets found for {token_a}/{token_b} pair")
            return {}

        # Fetch data for each market
        markets_data = {}
        for market_config in available_markets:
            market_data = fetch_token_pair_market_data(market_config, limit)
            if market_data:
                market_key = market_config['market_key']
                markets_data[market_key] = market_data

        return markets_data

    except Exception as e:
        st.error(f"Error fetching multi-market data: {str(e)}")
        return {}

def calculate_token_pair_strategy_apy(market_data: Dict, leverage: float, time_period_hours: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Calculate strategy APY for a token pair market

    Args:
        market_data (Dict): Market data including rates and staking
        leverage (float): Leverage multiplier
        time_period_hours (Optional[int]): Number of hours to filter data

    Returns:
        pd.DataFrame: Strategy calculation results
    """
    try:
        # Extract data
        token_a_rates = market_data['rates_data']['token_a_rates']
        token_b_rates = market_data['rates_data']['token_b_rates']
        token_a_staking = market_data['staking_data']['token_a_staking']
        token_b_staking = market_data['staking_data']['token_b_staking']
        metadata = market_data['metadata']

        # Use existing calculate_net_apy function with proper structure
        strategy_data = {
            'long_staking': token_a_staking,
            'short_staking': token_b_staking,
            'long_rates': token_a_rates,
            'short_rates': token_b_rates
        }

        return calculate_net_apy(
            strategy_data,
            leverage,
            metadata['token_a_symbol'],
            metadata['token_b_symbol'],
            time_period_hours
        )

    except Exception as e:
        st.error(f"Error calculating strategy APY: {str(e)}")
        return None

def calculate_multi_market_fees(markets_data: Dict, principal_amount: float, leverage: float, time_period_hours: int) -> Dict:
    """
    Calculate fees for all available markets

    Args:
        markets_data (Dict): Data for all markets
        principal_amount (float): Principal amount
        leverage (float): Leverage multiplier
        time_period_hours (int): Time period in hours

    Returns:
        Dict: Fee calculations for all markets
    """
    markets_fees = {}
    fee_structure = STRATEGY_CONFIG['asgard_spot']['fee_structure']
    position_size = principal_amount * leverage

    for market_key, market_data in markets_data.items():
        try:
            # Calculate strategy APY
            strategy_df = calculate_token_pair_strategy_apy(market_data, leverage, time_period_hours)

            if strategy_df is None or strategy_df.empty:
                continue

            # Calculate average net APY over time period
            avg_net_apy = strategy_df['net_apy'].mean()

            # Variable fees = negative of net APY (since APY is earned, fees are paid)
            variable_fees_rate = -avg_net_apy

            # Convert to hourly rate and calculate total variable fees
            hourly_variable_rate = variable_fees_rate / HOURS_PER_YEAR
            total_variable_fees = (hourly_variable_rate / 100) * position_size * time_period_hours

            # Fixed fees
            opening_fee = position_size * fee_structure['opening_rate']
            closing_fee = position_size * fee_structure['closing_rate']
            total_fixed_fees = opening_fee + closing_fee

            # Total fees
            total_fees = total_fixed_fees + total_variable_fees

            markets_fees[market_key] = {
                'opening_fee': opening_fee,
                'closing_fee': closing_fee,
                'total_fixed_fees': total_fixed_fees,
                'avg_net_apy': avg_net_apy,
                'variable_fees_rate': variable_fees_rate,
                'total_variable_fees': total_variable_fees,
                'total_fees': total_fees,
                'position_size': position_size,
                'strategy_df': strategy_df,
                'metadata': market_data['metadata']
            }

        except Exception as e:
            st.error(f"Error calculating fees for {market_key}: {str(e)}")
            continue

    return markets_fees

def plot_combined_fee_rates_chart(sol_fees_df: pd.DataFrame, asgard_strategy_df: pd.DataFrame, leverage: float, jupiter_available: bool, asgard_available: bool) -> Optional[go.Figure]:
    """
    Create a combined chart showing both Jupiter Perps borrow rates and Asgard net APY

    Args:
        sol_fees_df (pd.DataFrame): DataFrame with Jupiter Perps fee calculations
        asgard_strategy_df (pd.DataFrame): DataFrame with Asgard strategy calculations
        leverage (float): Leverage multiplier
        jupiter_available (bool): Whether Jupiter data is available
        asgard_available (bool): Whether Asgard data is available

    Returns:
        plotly.graph_objects.Figure: Chart figure or None if error
    """
    fig = go.Figure()

    # Add Jupiter Perps hourly borrow rate trace if available
    if jupiter_available and sol_fees_df is not None and not sol_fees_df.empty:
        fig.add_trace(go.Scatter(
            x=sol_fees_df['hourBucket'],
            y=sol_fees_df['hourly_borrow_rate_percent'],
            mode='lines+markers',
            name='Jupiter Perps - Hourly Borrow Rate',
            line=dict(color='#FF6B35', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Jupiter Perps</b><br>' +
                          'Time: %{x}<br>' +
                          'Hourly Rate: %{y:.6f}%<br>' +
                          '<extra></extra>'
        ))

    # Add Asgard equivalent hourly borrow rate trace if available
    if asgard_available and asgard_strategy_df is not None and not asgard_strategy_df.empty:
        # Convert net APY to equivalent hourly borrow rate
        # Negative because positive APY = negative borrow cost (you earn money)
        asgard_hourly_rate = -asgard_strategy_df['net_apy'] / (365 * 24)

        fig.add_trace(go.Scatter(
            x=asgard_strategy_df['hourBucket'],
            y=asgard_hourly_rate,
            mode='lines+markers',
            name='Asgard - Equivalent Hourly Rate',
            line=dict(color='#2E8B57', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Asgard Strategy</b><br>' +
                          'Time: %{x}<br>' +
                          'Hourly Rate: %{y:.6f}%<br>' +
                          'Net APY: %{customdata:.4f}%<br>' +
                          '<extra></extra>',
            customdata=asgard_strategy_df['net_apy']
        ))

        # Update layout - single axis since both are now hourly rates
    if jupiter_available and asgard_available:
        title = 'Hourly Borrow Rates Comparison'
    elif jupiter_available:
        title = 'Jupiter Perps - Hourly Borrow Rates'
    else:
        title = 'Asgard - Equivalent Hourly Rates'

    fig.update_layout(
        title=f'{title} (Leverage: {leverage}x)',
        xaxis_title='Time',
        yaxis_title='Hourly Rate (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Format y-axis to show percentage with 6 decimal places
    fig.update_yaxes(tickformat='.6f')

    # Add horizontal line at 0% for reference
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="Break-even line", annotation_position="bottom right")

    return fig

def plot_multi_strategy_comparison_chart(
    jupiter_data: Optional[pd.DataFrame],
    drift_data: Optional[pd.DataFrame],
    asgard_markets_fees: Dict,
    leverage: float
) -> Optional[go.Figure]:
    """
    Create a chart showing Jupiter Perps, Drift Perps, and all Asgard markets on the same hourly rate scale

    Args:
        jupiter_data (Optional[pd.DataFrame]): Jupiter Perps fee data
        drift_data (Optional[pd.DataFrame]): Drift Perps fee data
        asgard_markets_fees (Dict): Fee calculations for all Asgard markets
        leverage (float): Leverage multiplier

    Returns:
        plotly.graph_objects.Figure: Multi-strategy comparison chart
    """
    fig = go.Figure()

    # Add Jupiter Perps trace if available
    if jupiter_data is not None and not jupiter_data.empty:
        fig.add_trace(go.Scatter(
            x=jupiter_data['hourBucket'],
            y=jupiter_data['hourly_borrow_rate_percent'],
            mode='lines+markers',
            name='Jupiter Perps (SOL)',
            line=dict(color=CHART_COLORS[0], width=2),  # Orange
            marker=dict(size=4),
            hovertemplate='<b>Jupiter Perps</b><br>' +
                          'Time: %{x}<br>' +
                          'Hourly Rate: %{y:.6f}%<br>' +
                          '<extra></extra>'
        ))

    # Add Drift Perps trace if available
    if drift_data is not None and not drift_data.empty:
        fig.add_trace(go.Scatter(
            x=drift_data['hourBucket'],
            y=drift_data['hourly_funding_rate_percent'],
            mode='lines+markers',
            name='Drift Perps (SOL)',
            line=dict(color=CHART_COLORS[1], width=2),  # Green
            marker=dict(size=4),
            hovertemplate='<b>Drift Perps</b><br>' +
                          'Time: %{x}<br>' +
                          'Hourly Rate: %{y:.6f}%<br>' +
                          '<extra></extra>'
        ))

    # Add Asgard market traces
    color_index = 2  # Start from index 2 since 0=Jupiter, 1=Drift
    for market_key, market_fees in asgard_markets_fees.items():
        if market_fees.get('strategy_df') is not None and not market_fees['strategy_df'].empty:
            strategy_df = market_fees['strategy_df']
            metadata = market_fees['metadata']

            # Convert net APY to equivalent hourly borrow rate
            asgard_hourly_rate = -strategy_df['net_apy'] / HOURS_PER_YEAR

            # Create readable market name
            market_name = f"Asgard ({metadata['token_a_symbol']}/{metadata['token_b_symbol']} - {metadata['protocol']})"

            fig.add_trace(go.Scatter(
                x=strategy_df['hourBucket'],
                y=asgard_hourly_rate,
                mode='lines+markers',
                name=market_name,
                line=dict(color=CHART_COLORS[color_index % len(CHART_COLORS)], width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{market_name}</b><br>' +
                              'Time: %{x}<br>' +
                              'Hourly Rate: %{y:.6f}%<br>' +
                              'Net APY: %{customdata:.4f}%<br>' +
                              '<extra></extra>',
                customdata=strategy_df['net_apy']
            ))
            color_index += 1

    # Determine chart title
    jupiter_available = jupiter_data is not None and not jupiter_data.empty
    drift_available = drift_data is not None and not drift_data.empty
    asgard_count = len([k for k, v in asgard_markets_fees.items()
                      if v.get('strategy_df') is not None and not v['strategy_df'].empty])

    perps_count = sum([jupiter_available, drift_available])
    total_strategies = perps_count + asgard_count

    if total_strategies > 1:
        title = 'Multi-Strategy Hourly Rates Comparison'
    elif jupiter_available:
        title = 'Jupiter Perps - Hourly Borrow Rates'
    elif drift_available:
        title = 'Drift Perps - Hourly Funding Rates'
    elif asgard_count > 0:
        title = 'Asgard Strategies - Equivalent Hourly Rates'
    else:
        title = 'Fee Rates Comparison'

    # Update layout
    fig.update_layout(
        title=f'{title} (Leverage: {leverage}x)',
        xaxis_title='Time',
        yaxis_title='Hourly Rate (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Format y-axis to show percentage with 6 decimal places
    fig.update_yaxes(tickformat='.6f')

    # Add horizontal line at 0% for reference
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="Break-even line", annotation_position="bottom right")

    return fig

# Streamlit App Layout
st.set_page_config(
    page_title="Spot vs Perps - Fee Analysis",
    page_icon="💰",
    layout="wide"
)

st.title("Spot vs Perps")

# Load token configuration
token_config = load_token_config()

if token_config:

    # SOL Fee Rates Section
    st.divider()
    st.subheader("💰 SOL Fee Rates Analysis")
    st.write("Analyze the complete fee structure for leveraged SOL trading on Jupiter Perpetuals.")

    # Fee analysis parameters
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        principal_amount = st.number_input(
            "Principal Amount ($):",
            min_value=100.0,
            max_value=10000000.0,
            value=100000.0,
            step=100.0,
            key="sol_principal_amount"
        )

    with col2:
        sol_leverage = st.number_input(
            "Leverage:",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="sol_leverage"
        )

    with col3:
        sol_fee_time_period = st.selectbox(
            "Time period:",
            options=[
                ("Last 30 Days", 720),
                ("Last 15 Days", 360),
                ("Last 7 Days", 168),
                ("Last 5 Days", 120),
                ("Last 3 Days", 72),
                ("Last Day", 24),
                ("Last 12 Hours", 12)
            ],
            format_func=lambda x: x[0],
            index=3,
            key="sol_fee_time_period"
        )

    # Calculate position size
    position_size = principal_amount * sol_leverage

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Principal Amount", f"${principal_amount:,.2f}")
    with col2:
        st.metric("Leverage", f"{sol_leverage}x")
    with col3:
        st.metric("Position Size", f"${position_size:,.2f}")

    # Auto-trigger analysis when parameters change
    analysis_key = f"{principal_amount}_{sol_leverage}_{sol_fee_time_period[1]}"

    # Initialize session state for tracking analysis
    if "last_analysis_key" not in st.session_state:
        st.session_state.last_analysis_key = None

    # Check if parameters have changed or this is the first load
    should_analyze = (
        st.session_state.last_analysis_key != analysis_key or
        st.session_state.last_analysis_key is None
    )

    if should_analyze:
        st.session_state.last_analysis_key = analysis_key

        # Add manual refresh option
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh Analysis", type="secondary", key="refresh_analysis_btn"):
                st.session_state.last_analysis_key = None
                st.session_state.force_refresh = True
                st.rerun()
        with col2:
            st.caption("Analysis runs automatically when parameters change. Click refresh to update data manually.")

    # Run analysis automatically or when requested
    if should_analyze or st.session_state.get("force_refresh", False):
        if st.session_state.get("force_refresh", False):
            st.session_state.force_refresh = False
        with st.spinner("Fetching fee data and calculating costs for all platforms..."):
            time_period_hours = sol_fee_time_period[1]

                        # Calculate timestamps for Drift API (round to 3 decimal places)
            from datetime import datetime
            end_time = round(datetime.now().timestamp(), 3)
            start_time = round(end_time - (time_period_hours * 3600), 3)

            # Fetch Jupiter perps data for SOL
            jupiter_perps_data = fetch_jupiter_perps_data("SOL", limit=time_period_hours)

            # Fetch Drift perps data for SOL (market index 0)
            drift_perps_data = fetch_drift_perps_data(0, start_time, end_time)

            # Fetch Asgard strategy data
            asgard_data = fetch_all_asgard_markets_data(principal_amount, sol_leverage, limit=time_period_hours)

            jupiter_success = jupiter_perps_data is not None
            drift_success = drift_perps_data is not None
            asgard_success = asgard_data and len(asgard_data) > 0  # Check if we have any market data

            if jupiter_success or drift_success or asgard_success:
                jupiter_fees_calculated = False
                drift_fees_calculated = False
                asgard_fees_calculated = False

                # Calculate Jupiter Perps fees if data available
                if jupiter_success:
                    sol_fees_df = calculate_sol_fees(jupiter_perps_data, principal_amount, sol_leverage, time_period_hours)
                    if sol_fees_df is not None and not sol_fees_df.empty:
                        jupiter_fees_calculated = True

                # Calculate Drift Perps fees if data available
                if drift_success:
                    drift_fees_df = calculate_drift_fees(drift_perps_data, principal_amount, sol_leverage, time_period_hours)
                    if drift_fees_df is not None and not drift_fees_df.empty:
                        drift_fees_calculated = True

                # Calculate Asgard fees if data available
                if asgard_success:
                    asgard_fees = calculate_multi_market_fees(asgard_data, principal_amount, sol_leverage, time_period_hours)
                    if asgard_fees:
                        asgard_fees_calculated = True

                if jupiter_fees_calculated or drift_fees_calculated or asgard_fees_calculated:
                    # Plot multi-strategy comparison chart
                    fig = plot_multi_strategy_comparison_chart(
                        sol_fees_df if jupiter_fees_calculated else None,
                        drift_fees_df if drift_fees_calculated else None,
                        asgard_fees if asgard_fees_calculated else {},
                        sol_leverage
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                                        # Fees comparison table
                    st.markdown("#### 📋 Fees Comparison")

                    # Initialize table data
                    positions = []
                    opening_fees = []
                    variable_fees = []
                    close_fees = []
                    total_fees = []

                    # Jupiter Perps calculations (if available)
                    if jupiter_fees_calculated:
                        jupiter_opening_fee = position_size * 0.0006  # 0.06%
                        jupiter_closing_fee = position_size * 0.0006  # 0.06%
                        avg_hourly_rate = sol_fees_df['hourly_borrow_rate_percent'].mean()
                        jupiter_variable_fees = (avg_hourly_rate / 100) * position_size * time_period_hours
                        jupiter_total_fees = jupiter_opening_fee + jupiter_closing_fee + jupiter_variable_fees

                        positions.append("Jupiter Perps (SOL)")
                        opening_fees.append(f"${jupiter_opening_fee:.2f}")
                        variable_fees.append(f"${jupiter_variable_fees:.2f}")
                        close_fees.append(f"${jupiter_closing_fee:.2f}")
                        total_fees.append(f"${jupiter_total_fees:.2f}")

                    # Drift Perps calculations (if available)
                    if drift_fees_calculated:
                        drift_opening_fee = position_size * 0.00025  # 0.0250%
                        drift_closing_fee = position_size * 0.00025  # 0.0250%
                        avg_funding_rate = drift_fees_df['hourly_funding_rate_percent'].mean()
                        drift_variable_fees = (avg_funding_rate / 100) * position_size * time_period_hours
                        drift_total_fees = drift_opening_fee + drift_closing_fee + drift_variable_fees

                        positions.append("Drift Perps (SOL)")
                        opening_fees.append(f"${drift_opening_fee:.2f}")
                        variable_fees.append(f"${drift_variable_fees:.2f}")
                        close_fees.append(f"${drift_closing_fee:.2f}")
                        total_fees.append(f"${drift_total_fees:.2f}")

                    # Asgard calculations (if available)
                    if asgard_fees_calculated:
                        for market_key, market_fees in asgard_fees.items():
                            metadata = market_fees['metadata']
                            market_name = f"Asgard ({metadata['token_a_symbol']}/{metadata['token_b_symbol']} - {metadata['protocol']})"

                            positions.append(market_name)
                            opening_fees.append(f"${market_fees['opening_fee']:.2f}")
                            variable_fees.append(f"${market_fees['total_variable_fees']:.2f}")
                            close_fees.append(f"${market_fees['closing_fee']:.2f}")
                            total_fees.append(f"${market_fees['total_fees']:.2f}")

                    # Create fees comparison table
                    fees_comparison = {
                        "Position": positions,
                        "Opening Fees": opening_fees,
                        "Variable Fees": variable_fees,
                        "Close Fees": close_fees,
                        "Total Fees": total_fees
                    }

                    fees_df = pd.DataFrame(fees_comparison)
                    st.dataframe(fees_df, use_container_width=True, hide_index=True)

                    # Show calculation details
                    with st.expander("🧮 Calculation Details"):
                        calculation_details = ""

                        if jupiter_fees_calculated:
                            calculation_details += f"""
## Jupiter Perps (SOL) Calculations:

**Fixed Fees:**
- Opening Fee = Position Size × 0.06% = ${position_size:,.2f} × 0.0006 = ${jupiter_opening_fee:.2f}
- Closing Fee = Position Size × 0.06% = ${position_size:,.2f} × 0.0006 = ${jupiter_closing_fee:.2f}
- Total Fixed Fees = ${jupiter_opening_fee + jupiter_closing_fee:.2f}

**Variable Fees:**
- Average Hourly Borrow Rate = {avg_hourly_rate:.6f}% per hour
- Time Period = {time_period_hours} hours ({sol_fee_time_period[0]})
- Total Variable Fees = Position Size × Avg Rate × Hours
- Total Variable Fees = ${position_size:,.2f} × {avg_hourly_rate:.6f}% × {time_period_hours} = ${jupiter_variable_fees:.2f}

**Jupiter Perps Total: ${jupiter_total_fees:.2f}**
"""

                        if drift_fees_calculated:
                            if jupiter_fees_calculated:
                                calculation_details += "\n---\n"

                            calculation_details += f"""
## Drift Perps (SOL) Calculations:

**Fixed Fees:**
- Opening Fee = Position Size × 0.0250% = ${position_size:,.2f} × 0.00025 = ${drift_opening_fee:.2f}
- Closing Fee = Position Size × 0.0250% = ${position_size:,.2f} × 0.00025 = ${drift_closing_fee:.2f}
- Total Fixed Fees = ${drift_opening_fee + drift_closing_fee:.2f}

**Variable Fees (Funding):**
- Funding Rate Formula: (fundingRate / 1e9) / (oraclePriceTwap / 1e6) × 100
- Average Hourly Funding Rate = {avg_funding_rate:.6f}% per hour
- Time Period = {time_period_hours} hours ({sol_fee_time_period[0]})
- Total Variable Fees = Position Size × Avg Rate × Hours
- Total Variable Fees = ${position_size:,.2f} × {avg_funding_rate:.6f}% × {time_period_hours} = ${drift_variable_fees:.2f}

**Drift Perps Total: ${drift_total_fees:.2f}**
**Note: Funding rates can be negative (profitable) or positive (cost)**
"""

                        if asgard_fees_calculated:
                            for market_key, market_fees in asgard_fees.items():
                                if jupiter_fees_calculated or drift_fees_calculated:
                                    calculation_details += "\n---\n"

                                metadata = market_fees['metadata']
                                equivalent_hourly_rate = -market_fees['avg_net_apy'] / HOURS_PER_YEAR

                                calculation_details += f"""
## Asgard ({metadata['token_a_symbol']}/{metadata['token_b_symbol']} - {metadata['protocol']}) Calculations:

**Fixed Fees:**
- Opening Fee = Position Size × {STRATEGY_CONFIG['asgard_spot']['fee_structure']['opening_rate']*100:.1f}% = ${position_size:,.2f} × {STRATEGY_CONFIG['asgard_spot']['fee_structure']['opening_rate']} = ${market_fees['opening_fee']:.2f}
- Closing Fee = Position Size × {STRATEGY_CONFIG['asgard_spot']['fee_structure']['closing_rate']*100:.1f}% = ${position_size:,.2f} × {STRATEGY_CONFIG['asgard_spot']['fee_structure']['closing_rate']} = ${market_fees['closing_fee']:.2f}
- Total Fixed Fees = ${market_fees['total_fixed_fees']:.2f}

**Variable Fees (Strategy Performance):**
- Average Net APY = {market_fees['avg_net_apy']:.4f}% per year
- Equivalent Hourly Rate = -{market_fees['avg_net_apy']:.4f}% ÷ (365 × 24) = {equivalent_hourly_rate:.6f}% per hour
- Variable Fees Rate = -{market_fees['avg_net_apy']:.4f}% = {market_fees['variable_fees_rate']:.4f}% per year
- Time Period = {time_period_hours} hours ({sol_fee_time_period[0]})
- Total Variable Fees = Position Size × Variable Rate × (Hours ÷ 8760)
- Total Variable Fees = ${position_size:,.2f} × {market_fees['variable_fees_rate']:.4f}% × ({time_period_hours} ÷ 8760) = ${market_fees['total_variable_fees']:.2f}

**Asgard Total: ${market_fees['total_fees']:.2f}**
**Chart Rate: {equivalent_hourly_rate:.6f}% per hour (negative = profitable strategy)**
"""

                        calculation_details += f"""

---

**Position Details:**
- Principal: ${principal_amount:,.2f}
- Leverage: {sol_leverage}x
- Position Size: ${position_size:,.2f}
- Time Period: {sol_fee_time_period[0]} ({time_period_hours} hours)

**Data Consistency:**
- All platforms now use the exact same time period: {sol_fee_time_period[0]}
- Data is filtered to only include records within the selected time window
- This ensures fair comparison across all platforms with consistent timeframes

**Key Differences:**
- **Jupiter Perps**: Direct hourly borrow rate (always positive cost)
- **Drift Perps**: Funding rate (can be positive cost or negative profit)
- **Asgard**: Equivalent hourly rate from net APY (negative = profitable, positive = unprofitable)
- **Chart**: All lines on same scale for direct comparison. Negative rates mean profitable strategies!
"""

                        st.markdown(calculation_details)

                    # Raw data
                    if jupiter_fees_calculated:
                        with st.expander("View Raw Fee Data (Jupiter Perps)"):
                            display_columns = [
                                'hourBucket', 'hourly_borrow_rate_percent', 'hourly_borrow_cost'
                            ]
                            display_df = sol_fees_df[display_columns].copy()
                            display_df.columns = [
                                'Time', 'Hourly Rate (%)', 'Hourly Cost ($)'
                            ]
                            st.dataframe(display_df, use_container_width=True)

                    if drift_fees_calculated:
                        with st.expander("View Raw Fee Data (Drift Perps)"):
                            display_columns = [
                                'hourBucket', 'hourly_funding_rate_percent', 'hourly_funding_cost'
                            ]
                            display_df = drift_fees_df[display_columns].copy()
                            display_df.columns = [
                                'Time', 'Hourly Rate (%)', 'Hourly Cost ($)'
                            ]
                            st.dataframe(display_df, use_container_width=True)

                    if asgard_fees_calculated:
                        for market_key, market_fees in asgard_fees.items():
                            if market_fees.get('strategy_df') is not None and not market_fees['strategy_df'].empty:
                                metadata = market_fees['metadata']
                                market_name = f"Asgard ({metadata['token_a_symbol']}/{metadata['token_b_symbol']} - {metadata['protocol']})"

                                with st.expander(f"View Raw Strategy Data ({market_name})"):
                                    strategy_df = market_fees['strategy_df']
                                    display_columns = [
                                        'hourBucket', 'net_apy', 'long_overall_apy', 'short_overall_apy'
                                    ]
                                    display_df = strategy_df[display_columns].copy()
                                    display_df.columns = [
                                        'Time', 'Net APY (%)', f'{metadata["token_a_symbol"]} APY (%)', f'{metadata["token_b_symbol"]} APY (%)'
                                    ]
                                    st.dataframe(display_df, use_container_width=True)
            else:
                st.error("Failed to fetch or calculate fee data. Please check your token configuration and API connectivity.")
