import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Load configuration files
@st.cache_data
def load_token_config() -> Dict:
    """Load token configuration from JSON file"""
    try:
        with open('token_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("token_config.json file not found")
        return {}
    except json.JSONDecodeError:
        st.error("Error parsing token_config.json")
        return {}

@st.cache_data
def load_url_config() -> Dict:
    """Load URL configuration from JSON file"""
    try:
        with open('url_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("url_config.json file not found")
        return {}
    except json.JSONDecodeError:
        st.error("Error parsing url_config.json")
        return {}

def fetch_staking_data(mint_address: str, limit: int = 168) -> Optional[Dict]:
    """
    Fetch staking APY data for a given mint address

    Args:
        mint_address (str): The mint address of the token
        limit (int): Number of hours of data to fetch (default: 168 = 1 week)

    Returns:
        Dict: API response data or None if error
    """
    url_config = load_url_config()
    base_url = url_config.get('staking_apy_hourly_base_url')

    if not base_url:
        st.error("Staking APY URL not configured")
        return None

    # Construct API URL with parameters
    api_url = f"{base_url}/{mint_address}?limit={limit}"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Error parsing API response")
        return None

def fetch_rates_data(bank_address: str, protocol: str, limit: int = 168) -> Optional[Dict]:
    """
    Fetch lending and borrowing rates data for a given bank address and protocol

    Args:
        bank_address (str): The bank address
        protocol (str): The protocol name
        limit (int): Number of hours of data to fetch (default: 168 = 1 week)

    Returns:
        Dict: API response data or None if error
    """
    url_config = load_url_config()
    base_url = url_config.get('spot_rates_hourly_base_url')

    if not base_url:
        st.error("Rates URL not configured")
        return None

    # Construct API URL with parameters
    api_url = f"{base_url}/{bank_address}/{protocol}?limit={limit}"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching rates data: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Error parsing API response")
        return None

def plot_staking_apy_chart(data: Dict, token_symbol: str) -> Optional[go.Figure]:
    """
    Create a line chart for staking APY data

    Args:
        data (Dict): API response data
        token_symbol (str): Token symbol for chart title

    Returns:
        plotly.graph_objects.Figure: Chart figure or None if error
    """
    if not data or not data.get('success'):
        st.error("No valid data to plot")
        return None

    records = data.get('data', {}).get('records', [])

    if not records:
        st.error("No records found in data")
        return None

    # Convert records to DataFrame
    df = pd.DataFrame(records)

    # Convert hourBucket to datetime and sort by time
    df['hourBucket'] = pd.to_datetime(df['hourBucket'])
    df = df.sort_values('hourBucket')

    # Convert APY from decimal to percentage
    df['avgApy_percent'] = df['avgApy'] * 100

    # Create the line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['hourBucket'],
        y=df['avgApy_percent'],
        mode='lines+markers',
        name=f'{token_symbol} APY',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>%{text}</b><br>' +
                      'Time: %{x}<br>' +
                      'APY: %{y:.4f}%<br>' +
                      '<extra></extra>',
        text=[token_symbol] * len(df)
    ))

    # Update layout
    fig.update_layout(
        title=f'{token_symbol} Staking APY Over Time',
        xaxis_title='Time',
        yaxis_title='APY (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )

    # Format y-axis to show percentage with 4 decimal places
    fig.update_yaxes(tickformat='.4f')

    return fig

def plot_lending_borrowing_rates_chart(data: Dict, token_symbol: str, protocol: str) -> Optional[go.Figure]:
    """
    Create a line chart for lending and borrowing rates data

    Args:
        data (Dict): API response data
        token_symbol (str): Token symbol for chart title
        protocol (str): Protocol name for chart title

    Returns:
        plotly.graph_objects.Figure: Chart figure or None if error
    """
    if not data or not data.get('success'):
        st.error("No valid data to plot")
        return None

    records = data.get('data', {}).get('records', [])

    if not records:
        st.error("No records found in data")
        return None

    # Convert records to DataFrame
    df = pd.DataFrame(records)

    # Convert hourBucket to datetime and sort by time
    df['hourBucket'] = pd.to_datetime(df['hourBucket'])
    df = df.sort_values('hourBucket')

    # Use rates as they are (already in percentage)
    df['avgLendingRate_percent'] = df['avgLendingRate']
    df['avgBorrowingRate_percent'] = df['avgBorrowingRate']

    # Create the line chart with both rates
    fig = go.Figure()

    # Add lending rate trace
    fig.add_trace(go.Scatter(
        x=df['hourBucket'],
        y=df['avgLendingRate_percent'],
        mode='lines+markers',
        name=f'Lending Rate',
        line=dict(color='#2E8B57', width=2),  # Sea green
        marker=dict(size=4),
        hovertemplate='<b>Lending Rate</b><br>' +
                      'Time: %{x}<br>' +
                      'Rate: %{y:.4f}%<br>' +
                      '<extra></extra>'
    ))

    # Add borrowing rate trace
    fig.add_trace(go.Scatter(
        x=df['hourBucket'],
        y=df['avgBorrowingRate_percent'],
        mode='lines+markers',
        name=f'Borrowing Rate',
        line=dict(color='#DC143C', width=2),  # Crimson
        marker=dict(size=4),
        hovertemplate='<b>Borrowing Rate</b><br>' +
                      'Time: %{x}<br>' +
                      'Rate: %{y:.4f}%<br>' +
                      '<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f'{token_symbol} Lending & Borrowing Rates - {protocol.title()}',
        xaxis_title='Time',
        yaxis_title='Rate (%)',
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

    # Format y-axis to show percentage with 4 decimal places
    fig.update_yaxes(tickformat='.4f')

    return fig

def fetch_strategy_data(long_token_info: Dict, short_token_info: Dict, long_bank: Dict, short_bank: Dict, protocol: str, limit: int = 168) -> Optional[Dict]:
    """
    Fetch combined staking and rates data for strategy calculation

    Args:
        long_token_info (Dict): Long token configuration
        short_token_info (Dict): Short token configuration
        long_bank (Dict): Long token bank info
        short_bank (Dict): Short token bank info
        protocol (str): Protocol name
        limit (int): Number of hours of data to fetch

    Returns:
        Dict: Combined data for strategy calculation or None if error
    """
    try:
        # Fetch staking data for long token (if it has staking yield)
        long_staking_data = None
        if long_token_info.get('hasStakingYield', False):
            long_staking_data = fetch_staking_data(long_token_info['mint'], limit)

        # Fetch staking data for short token (if it has staking yield)
        short_staking_data = None
        if short_token_info.get('hasStakingYield', False):
            short_staking_data = fetch_staking_data(short_token_info['mint'], limit)

        # Fetch rates data for both tokens
        long_rates_data = fetch_rates_data(long_bank['bank'], protocol, limit)
        short_rates_data = fetch_rates_data(short_bank['bank'], protocol, limit)

        if not long_rates_data or not short_rates_data:
            return None

        return {
            'long_staking': long_staking_data,
            'short_staking': short_staking_data,
            'long_rates': long_rates_data,
            'short_rates': short_rates_data
        }

    except Exception as e:
        st.error(f"Error fetching strategy data: {str(e)}")
        return None

def calculate_net_apy(strategy_data: Dict, leverage: float, long_token_symbol: str, short_token_symbol: str) -> Optional[pd.DataFrame]:
    """
    Calculate net APY for the strategy over time

    Args:
        strategy_data (Dict): Combined strategy data
        leverage (float): Leverage amount
        long_token_symbol (str): Long token symbol
        short_token_symbol (str): Short token symbol

    Returns:
        pd.DataFrame: DataFrame with net APY calculations or None if error
    """
    try:
        # Extract rates data
        long_rates_records = strategy_data['long_rates']['data']['records']
        short_rates_records = strategy_data['short_rates']['data']['records']

        # Convert to DataFrames
        long_rates_df = pd.DataFrame(long_rates_records)
        short_rates_df = pd.DataFrame(short_rates_records)

        # Convert timestamps and sort
        long_rates_df['hourBucket'] = pd.to_datetime(long_rates_df['hourBucket'])
        short_rates_df['hourBucket'] = pd.to_datetime(short_rates_df['hourBucket'])

        long_rates_df = long_rates_df.sort_values('hourBucket')
        short_rates_df = short_rates_df.sort_values('hourBucket')

        # Get staking data if available
        long_staking_df = None
        if strategy_data['long_staking'] and strategy_data['long_staking'].get('success'):
            long_staking_records = strategy_data['long_staking']['data']['records']
            if long_staking_records:
                long_staking_df = pd.DataFrame(long_staking_records)
                long_staking_df['hourBucket'] = pd.to_datetime(long_staking_df['hourBucket'])
                long_staking_df = long_staking_df.sort_values('hourBucket')

        short_staking_df = None
        if strategy_data['short_staking'] and strategy_data['short_staking'].get('success'):
            short_staking_records = strategy_data['short_staking']['data']['records']
            if short_staking_records:
                short_staking_df = pd.DataFrame(short_staking_records)
                short_staking_df['hourBucket'] = pd.to_datetime(short_staking_df['hourBucket'])
                short_staking_df = short_staking_df.sort_values('hourBucket')

        # Merge data on timestamp - use rates data as base since it's required
        # Start with long rates
        merged_df = long_rates_df[['hourBucket', 'avgLendingRate']].copy()
        merged_df = merged_df.rename(columns={'avgLendingRate': 'long_lending_rate'})

        # Add short borrowing rates
        short_rates_subset = short_rates_df[['hourBucket', 'avgBorrowingRate']].copy()
        short_rates_subset = short_rates_subset.rename(columns={'avgBorrowingRate': 'short_borrowing_rate'})
        merged_df = pd.merge(merged_df, short_rates_subset, on='hourBucket', how='inner')

        # Add staking data if available
        if long_staking_df is not None:
            long_staking_subset = long_staking_df[['hourBucket', 'avgApy']].copy()
            long_staking_subset = long_staking_subset.rename(columns={'avgApy': 'long_staking_apy'})
            merged_df = pd.merge(merged_df, long_staking_subset, on='hourBucket', how='left')
            # Forward fill missing staking data and convert from decimal to percentage
            merged_df['long_staking_apy'] = merged_df['long_staking_apy'].ffill() * 100
        else:
            merged_df['long_staking_apy'] = 0

        if short_staking_df is not None:
            short_staking_subset = short_staking_df[['hourBucket', 'avgApy']].copy()
            short_staking_subset = short_staking_subset.rename(columns={'avgApy': 'short_staking_apy'})
            merged_df = pd.merge(merged_df, short_staking_subset, on='hourBucket', how='left')
            # Forward fill missing staking data and convert from decimal to percentage
            merged_df['short_staking_apy'] = merged_df['short_staking_apy'].ffill() * 100
        else:
            merged_df['short_staking_apy'] = 0

        # Fill any remaining NaN values with 0
        merged_df = merged_df.fillna(0)

        # Calculate overall APYs (lending/borrowing rates already in %, staking now in %)
        merged_df['long_overall_apy'] = (merged_df['long_lending_rate'] + merged_df['long_staking_apy'])
        merged_df['short_overall_apy'] = (merged_df['short_borrowing_rate'] + merged_df['short_staking_apy'])

        # Calculate net APY using the formula
        # Net APY = (Long Overall APY * Leverage) - (Short Overall APY * (Leverage - 1))
        merged_df['net_apy'] = (merged_df['long_overall_apy'] * leverage) - (merged_df['short_overall_apy'] * (leverage - 1))

        # Add metadata
        merged_df['long_token'] = long_token_symbol
        merged_df['short_token'] = short_token_symbol
        merged_df['leverage'] = leverage

        return merged_df

    except Exception as e:
        st.error(f"Error calculating net APY: {str(e)}")
        return None

def plot_strategy_chart(strategy_df: pd.DataFrame, long_token: str, short_token: str, leverage: float) -> Optional[go.Figure]:
    """
    Create a chart showing the strategy's net APY over time

    Args:
        strategy_df (pd.DataFrame): Strategy calculation results
        long_token (str): Long token symbol
        short_token (str): Short token symbol
        leverage (float): Leverage amount

    Returns:
        plotly.graph_objects.Figure: Strategy chart or None if error
    """
    try:
        fig = go.Figure()

        # Add net APY trace
        fig.add_trace(go.Scatter(
            x=strategy_df['hourBucket'],
            y=strategy_df['net_apy'],
            mode='lines+markers',
            name=f'Net APY',
            line=dict(color='#FF6B35', width=3),  # Orange
            marker=dict(size=4),
            hovertemplate='<b>Net APY</b><br>' +
                          'Time: %{x}<br>' +
                          'Net APY: %{y:.2f}%<br>' +
                          '<extra></extra>'
        ))

        # Add reference lines for component APYs
        fig.add_trace(go.Scatter(
            x=strategy_df['hourBucket'],
            y=strategy_df['long_overall_apy'],
            mode='lines',
            name=f'{long_token} Overall APY',
            line=dict(color='#2E8B57', width=1, dash='dash'),  # Sea green
            opacity=0.7,
            hovertemplate=f'<b>{long_token} Overall APY</b><br>' +
                          'Time: %{x}<br>' +
                          'APY: %{y:.2f}%<br>' +
                          '<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=strategy_df['hourBucket'],
            y=strategy_df['short_overall_apy'],
            mode='lines',
            name=f'{short_token} Overall APY',
            line=dict(color='#DC143C', width=1, dash='dash'),  # Crimson
            opacity=0.7,
            hovertemplate=f'<b>{short_token} Overall APY</b><br>' +
                          'Time: %{x}<br>' +
                          'APY: %{y:.2f}%<br>' +
                          '<extra></extra>'
        ))

        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        # Update layout
        fig.update_layout(
            title=f'Strategy: Long {long_token} / Short {short_token} (Leverage: {leverage}x)',
            xaxis_title='Time',
            yaxis_title='APY (%)',
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

        # Format y-axis to show percentage
        fig.update_yaxes(tickformat='.2f')

        return fig

    except Exception as e:
        st.error(f"Error creating strategy chart: {str(e)}")
        return None

# Streamlit App Layout
st.sidebar.title("Spot vs Perps")

st.title("Spot vs Perps")
st.write(
    "This app compares the performance of spot and perpetual futures contracts."
)

# Load token configuration
token_config = load_token_config()

if token_config:
    # Staking APY Chart Section
    st.subheader("📈 Staking APY Analysis")
    st.write("Select a token to view its historical staking APY data.")

    # Token selection
    col1, col2 = st.columns([2, 1])

    with col1:
        # Filter tokens that have staking yield
        staking_tokens = [token for token, config in token_config.items()
                         if config.get('hasStakingYield', False)]

        if not staking_tokens:
            st.warning("No tokens with staking yield found in configuration.")
            selected_token = None
        else:
            selected_token = st.selectbox(
                "Choose a token:",
                options=staking_tokens,
                index=0
            )

    with col2:
        time_period = st.selectbox(
            "Time period:",
            options=[
                ("1 Month", 720),
                ("15 Days", 360),
                ("1 Week", 168),
                ("3 Days", 72),
                ("1 Day", 24),
                ("12 Hours", 12)
            ],
            format_func=lambda x: x[0],
            index=2
        )

    if selected_token and st.button("Fetch APY Data", type="primary"):
        token_info = token_config[selected_token]
        mint_address = token_info['mint']

        with st.spinner(f"Fetching {selected_token} staking data..."):
            # Fetch data
            staking_data = fetch_staking_data(mint_address, limit=time_period[1])

            if staking_data:
                # Plot chart
                fig = plot_staking_apy_chart(staking_data, selected_token)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # Show some basic statistics
                    records = staking_data.get('data', {}).get('records', [])
                    if records:
                        df = pd.DataFrame(records)
                        df['avgApy_percent'] = df['avgApy'] * 100

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Current APY", f"{df['avgApy_percent'].iloc[0]:.4f}%")

                        with col2:
                            st.metric("Average APY", f"{df['avgApy_percent'].mean():.4f}%")

                        with col3:
                            st.metric("Min APY", f"{df['avgApy_percent'].min():.4f}%")

                        with col4:
                            st.metric("Max APY", f"{df['avgApy_percent'].max():.4f}%")

                        # Show raw data (optional)
                        with st.expander("View Raw Data"):
                            st.dataframe(df[['hourBucket', 'avgApy_percent', 'sampleCount']], use_container_width=True)

    # Lending & Borrowing Rates Section
    st.divider()
    st.subheader("💰 Lending & Borrowing Rates")
    st.write("Select a token and protocol to view historical lending and borrowing rates.")

    # Token and protocol selection
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Get all tokens that have banks (lending/borrowing data)
        tokens_with_banks = [token for token, config in token_config.items()
                           if config.get('banks', [])]

        if not tokens_with_banks:
            st.warning("No tokens with lending/borrowing data found in configuration.")
            selected_rates_token = None
        else:
            selected_rates_token = st.selectbox(
                "Choose a token:",
                options=tokens_with_banks,
                index=0,
                key="rates_token_select"
            )

    with col2:
        if selected_rates_token:
            token_banks = token_config[selected_rates_token].get('banks', [])
            protocols = list(set([bank['protocol'] for bank in token_banks]))

            selected_protocol = st.selectbox(
                "Protocol:",
                options=protocols,
                index=0,
                key="rates_protocol_select"
            )
        else:
            selected_protocol = None

    with col3:
        rates_time_period = st.selectbox(
            "Time period:",
            options=[
                ("1 Month", 720),
                ("15 Days", 360),
                ("1 Week", 168),
                ("3 Days", 72),
                ("1 Day", 24),
                ("12 Hours", 12)
            ],
            format_func=lambda x: x[0],
            index=2,
            key="rates_time_period_select"
        )

    # Market selection based on selected token and protocol
    if selected_rates_token and selected_protocol:
        token_banks = token_config[selected_rates_token].get('banks', [])
        protocol_banks = [bank for bank in token_banks if bank['protocol'] == selected_protocol]

        if protocol_banks:
            if len(protocol_banks) > 1:
                col1, col2 = st.columns([2, 2])
                with col1:
                    selected_market = st.selectbox(
                        "Market:",
                        options=protocol_banks,
                        format_func=lambda x: x['market'],
                        index=0,
                        key="rates_market_select"
                    )
            else:
                selected_market = protocol_banks[0]
                st.info(f"Market: {selected_market['market']}")

            if st.button("Fetch Rates Data", type="primary", key="fetch_rates_btn"):
                bank_address = selected_market['bank']

                with st.spinner(f"Fetching {selected_rates_token} rates data from {selected_protocol}..."):
                    # Fetch rates data
                    rates_data = fetch_rates_data(bank_address, selected_protocol, limit=rates_time_period[1])

                    if rates_data:
                        # Plot chart
                        fig = plot_lending_borrowing_rates_chart(rates_data, selected_rates_token, selected_protocol)

                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                            # Show some basic statistics
                            records = rates_data.get('data', {}).get('records', [])
                            if records:
                                df = pd.DataFrame(records)
                                df['avgLendingRate_percent'] = df['avgLendingRate']
                                df['avgBorrowingRate_percent'] = df['avgBorrowingRate']

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Current Lending Rate", f"{df['avgLendingRate_percent'].iloc[0]:.4f}%")

                                with col2:
                                    st.metric("Current Borrowing Rate", f"{df['avgBorrowingRate_percent'].iloc[0]:.4f}%")

                                with col3:
                                    st.metric("Avg Lending Rate", f"{df['avgLendingRate_percent'].mean():.4f}%")

                                with col4:
                                    st.metric("Avg Borrowing Rate", f"{df['avgBorrowingRate_percent'].mean():.4f}%")

                                # Show additional metrics
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Min Lending Rate", f"{df['avgLendingRate_percent'].min():.4f}%")

                                with col2:
                                    st.metric("Max Lending Rate", f"{df['avgLendingRate_percent'].max():.4f}%")

                                with col3:
                                    st.metric("Min Borrowing Rate", f"{df['avgBorrowingRate_percent'].min():.4f}%")

                                with col4:
                                    st.metric("Max Borrowing Rate", f"{df['avgBorrowingRate_percent'].max():.4f}%")

                                # Show raw data (optional)
                                with st.expander("View Raw Data"):
                                    display_df = df[['hourBucket', 'avgLendingRate_percent', 'avgBorrowingRate_percent', 'sampleCount']].copy()
                                    display_df.columns = ['Time', 'Lending Rate (%)', 'Borrowing Rate (%)', 'Sample Count']
                                    st.dataframe(display_df, use_container_width=True)

    # Strategy Section
    st.divider()
    st.subheader("🎯 Strategy Builder")
    st.write("Create long/short strategies with leverage to analyze potential returns.")

    # Strategy configuration
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📈 Long Position")

        # Long token selection
        tokens_with_banks = [token for token, config in token_config.items()
                           if config.get('banks', [])]

        if not tokens_with_banks:
            st.warning("No tokens with lending/borrowing data found.")
            long_token = None
        else:
            long_token = st.selectbox(
                "Choose long token:",
                options=tokens_with_banks,
                index=0,
                key="strategy_long_token"
            )

        # Long protocol selection
        if long_token:
            long_token_banks = token_config[long_token].get('banks', [])
            long_protocols = list(set([bank['protocol'] for bank in long_token_banks]))

            long_protocol = st.selectbox(
                "Protocol:",
                options=long_protocols,
                index=0,
                key="strategy_long_protocol"
            )
        else:
            long_protocol = None

        # Long market selection
        if long_token and long_protocol:
            long_protocol_banks = [bank for bank in long_token_banks if bank['protocol'] == long_protocol]

            if len(long_protocol_banks) > 1:
                long_market = st.selectbox(
                    "Market:",
                    options=long_protocol_banks,
                    format_func=lambda x: x['market'],
                    index=0,
                    key="strategy_long_market"
                )
            else:
                long_market = long_protocol_banks[0]
                st.info(f"Market: {long_market['market']}")

    with col2:
        st.markdown("#### 📉 Short Position")

        # Short token selection (filtered by same protocol and market)
        if long_token and long_protocol and 'long_market' in locals():
            # Find all tokens that have the same protocol and market
            available_short_tokens = []
            for token_symbol, token_info in token_config.items():
                token_banks = token_info.get('banks', [])
                for bank in token_banks:
                    if (bank['protocol'] == long_protocol and
                        bank['market'] == long_market['market']):
                        available_short_tokens.append(token_symbol)
                        break

            # Remove duplicates and the long token itself
            available_short_tokens = list(set(available_short_tokens))
            if long_token in available_short_tokens:
                available_short_tokens.remove(long_token)

            if not available_short_tokens:
                st.warning(f"No other tokens available in {long_protocol} - {long_market['market']}")
                short_token = None
            else:
                short_token = st.selectbox(
                    "Choose short token:",
                    options=available_short_tokens,
                    index=0,
                    key="strategy_short_token"
                )

                # Find the corresponding bank for short token
                if short_token:
                    short_token_banks = token_config[short_token].get('banks', [])
                    short_market = None
                    for bank in short_token_banks:
                        if (bank['protocol'] == long_protocol and
                            bank['market'] == long_market['market']):
                            short_market = bank
                            break
                else:
                    short_market = None
        else:
            st.info("Select long token configuration first")
            short_token = None
            short_market = None

    # Strategy parameters
    st.markdown("#### ⚡ Strategy Parameters")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        leverage = st.number_input(
            "Leverage:",
            min_value=1.0,
            max_value=20.0,
            value=7.0,
            step=0.5,
            key="strategy_leverage"
        )

    with col2:
        strategy_time_period = st.selectbox(
            "Time period:",
            options=[
                ("1 Month", 720),
                ("15 Days", 360),
                ("1 Week", 168),
                ("3 Days", 72),
                ("1 Day", 24),
                ("12 Hours", 12)
            ],
            format_func=lambda x: x[0],
            index=2,
            key="strategy_time_period"
        )

    with col3:
        st.markdown("")  # Spacing
        calculate_strategy = st.button(
            "Calculate Strategy",
            type="primary",
            key="calculate_strategy_btn",
            disabled=not (long_token and short_token and 'long_market' in locals() and short_market)
        )

    # Display strategy calculation
    if calculate_strategy and long_token and short_token and 'long_market' in locals() and short_market:
        with st.spinner(f"Calculating strategy: Long {long_token} / Short {short_token} ({leverage}x)..."):

            # Get token info
            long_token_info = token_config[long_token]
            short_token_info = token_config[short_token]

            # Fetch strategy data
            strategy_data = fetch_strategy_data(
                long_token_info,
                short_token_info,
                long_market,
                short_market,
                long_protocol,
                strategy_time_period[1]
            )

            if strategy_data:
                # Calculate net APY
                strategy_df = calculate_net_apy(
                    strategy_data,
                    leverage,
                    long_token,
                    short_token
                )

                if strategy_df is not None and not strategy_df.empty:
                    # Plot strategy chart
                    fig = plot_strategy_chart(strategy_df, long_token, short_token, leverage)

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                        # Strategy metrics
                        st.markdown("#### 📊 Strategy Performance")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            current_net_apy = strategy_df['net_apy'].iloc[0]
                            st.metric("Current Net APY", f"{current_net_apy:.2f}%")

                        with col2:
                            avg_net_apy = strategy_df['net_apy'].mean()
                            st.metric("Average Net APY", f"{avg_net_apy:.2f}%")

                        with col3:
                            min_net_apy = strategy_df['net_apy'].min()
                            st.metric("Min Net APY", f"{min_net_apy:.2f}%")

                        with col4:
                            max_net_apy = strategy_df['net_apy'].max()
                            st.metric("Max Net APY", f"{max_net_apy:.2f}%")

                        # Component breakdown for latest data point
                        st.markdown("#### 🧮 Component Breakdown (Latest)")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"**{long_token} (Long Position)**")
                            long_lending = strategy_df['long_lending_rate'].iloc[0]
                            long_staking = strategy_df['long_staking_apy'].iloc[0]
                            long_total = strategy_df['long_overall_apy'].iloc[0]

                            st.write(f"• Lending APY: {long_lending:.4f}%")
                            st.write(f"• Staking APY: {long_staking:.4f}%")
                            st.write(f"• **Total APY: {long_total:.4f}%**")
                            st.write(f"• **Leveraged APY: {long_total * leverage:.2f}%**")

                        with col2:
                            st.markdown(f"**{short_token} (Short Position)**")
                            short_borrowing = strategy_df['short_borrowing_rate'].iloc[0]
                            short_staking = strategy_df['short_staking_apy'].iloc[0]
                            short_total = strategy_df['short_overall_apy'].iloc[0]

                            st.write(f"• Borrowing APY: {short_borrowing:.4f}%")
                            st.write(f"• Staking APY: {short_staking:.4f}%")
                            st.write(f"• **Total APY: {short_total:.4f}%**")
                            st.write(f"• **Cost APY: {short_total * (leverage - 1):.2f}%**")

                        # Calculation formula display
                        with st.expander("💡 Calculation Formula"):
                            st.markdown(f"""
                            **Net APY Calculation:**

                            ```
                            Long Token Overall APY = Lending APY + Staking APY
                            Short Token Overall APY = Borrowing APY + Staking APY

                            Net APY = (Long Overall APY × Leverage) - (Short Overall APY × (Leverage - 1))
                            ```

                            **For your strategy:**
                            ```
                            Long {long_token} APY = {long_lending:.4f}% + {long_staking:.4f}% = {long_total:.4f}%
                            Short {short_token} APY = {short_borrowing:.4f}% + {short_staking:.4f}% = {short_total:.4f}%

                            Net APY = ({long_total:.4f}% × {leverage}) - ({short_total:.4f}% × {leverage - 1})
                                    = {long_total * leverage:.2f}% - {short_total * (leverage - 1):.2f}%
                                    = {current_net_apy:.2f}%
                            ```
                            """)

                        # Raw data table
                        with st.expander("View Strategy Data"):
                            display_columns = [
                                'hourBucket', 'long_overall_apy', 'short_overall_apy',
                                'net_apy', 'long_lending_rate', 'long_staking_apy',
                                'short_borrowing_rate', 'short_staking_apy'
                            ]
                            display_df = strategy_df[display_columns].copy()
                            display_df.columns = [
                                'Time', 'Long Overall APY (%)', 'Short Overall APY (%)',
                                'Net APY (%)', 'Long Lending (%)', 'Long Staking (%)',
                                'Short Borrowing (%)', 'Short Staking (%)'
                            ]
                            st.dataframe(display_df, use_container_width=True)

st.subheader("Spot")
st.write("Spot is a contract for difference (CFD) that allows you to trade the underlying asset directly.")

st.subheader("Perps")
st.write("Perps are a type of perpetual futures contract that allows you to trade the underlying asset directly.")
