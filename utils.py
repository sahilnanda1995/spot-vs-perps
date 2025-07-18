import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json
from typing import Dict, List, Optional, Tuple

# Configuration constants for generic token pair system
STRATEGY_CONFIG = {
    'asgard_spot': {
        'token_a': 'JitoSOL',  # Long position token
        'token_b': 'USDC',     # Short position token
        'fee_structure': {
            'opening_rate': 0.001,  # 0.1%
            'closing_rate': 0.001   # 0.1%
        }
    }
}

CHART_COLORS = ['#FF6B35', '#2E8B57', '#1E90FF', '#8A2BE2']  # Orange, Green, Blue, Purple
HOURS_PER_YEAR = 365 * 24

def discover_token_pair_markets(token_config: Dict, token_a: str, token_b: str) -> List[Dict]:
    """
    Discover all markets where both tokenA and tokenB are available

    Args:
        token_config (Dict): Token configuration data
        token_a (str): First token symbol (long position)
        token_b (str): Second token symbol (short position)

    Returns:
        List[Dict]: Available market configurations
    """
    if token_a not in token_config or token_b not in token_config:
        return []

    token_a_info = token_config[token_a]
    token_b_info = token_config[token_b]

    # Find common markets
    available_markets = []

    for token_a_bank in token_a_info.get('banks', []):
        for token_b_bank in token_b_info.get('banks', []):
            if (token_a_bank['protocol'] == token_b_bank['protocol'] and
                token_a_bank['market'] == token_b_bank['market']):

                market_config = {
                    'protocol': token_a_bank['protocol'],
                    'market': token_a_bank['market'],
                    'token_a_bank': token_a_bank['bank'],
                    'token_b_bank': token_b_bank['bank'],
                    'token_a_info': token_a_info,
                    'token_b_info': token_b_info,
                    'market_key': f"{token_a_bank['protocol']}_{token_a_bank['market'].replace(' ', '_').lower()}"
                }
                available_markets.append(market_config)

    return available_markets

def get_token_overall_apy(base_rate: float, staking_data: Optional[Dict], token_symbol: str) -> float:
    """
    Calculate overall APY for any token including staking yield

    Args:
        base_rate (float): Base lending/borrowing rate
        staking_data (Optional[Dict]): Staking APY data if available
        token_symbol (str): Token symbol for logging

    Returns:
        float: Overall APY including staking component
    """
    if staking_data and staking_data.get('success'):
        records = staking_data.get('data', {}).get('records', [])
        if records:
            # Convert from decimal to percentage (avgApy is in decimal format)
            staking_apy = records[0]['avgApy'] * 100
            return base_rate + staking_apy

    return base_rate  # No staking data available

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

def fetch_jupiter_perps_data(token_symbol: str, limit: int = 168) -> Optional[Dict]:
    """
    Fetch Jupiter perpetuals borrow rates data for a given token

    Args:
        token_symbol (str): The token symbol (SOL, ETH, BTC)
        limit (int): Number of hours of data to fetch (default: 168 = 1 week)

    Returns:
        Dict: API response data or None if error
    """
    url_config = load_url_config()
    base_url = url_config.get('jupiter_perpetuals_hourly_base_url')

    if not base_url:
        st.error("Jupiter perpetuals URL not configured")
        return None

    # Construct API URL with parameters
    api_url = f"{base_url}/{token_symbol}?limit={limit}"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Jupiter perps data: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Error parsing API response")
        return None

def fetch_drift_perps_data(market_index: int, from_timestamp: float, to_timestamp: float) -> Optional[Dict]:
    """
    Fetch Drift perpetuals funding rates data for a given market

    Args:
        market_index (int): Market index (0=SOL, 1=BTC, 2=ETH)
        from_timestamp (float): Unix timestamp start
        to_timestamp (float): Unix timestamp end

    Returns:
        Dict: API response data or None if error
    """
    url_config = load_url_config()
    base_url = url_config.get('drift_perpetuals_base_url')

    if not base_url:
        st.error("Drift perpetuals URL not configured")
        return None

    # Construct API URL with parameters
    api_url = f"{base_url}?marketIndex={market_index}&from={from_timestamp}&to={to_timestamp}"

        # Set headers to mimic a browser request from app.drift.trade
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://app.drift.trade/',
        'Origin': 'https://app.drift.trade',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site'
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Validate response structure
        if data.get('status') != 'ok':
            st.error("Invalid response from Drift API")
            return None

        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Drift perps data: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Error parsing Drift API response")
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

def plot_jupiter_perps_borrow_rates_chart(data: Dict, token_symbol: str, leverage: float) -> Optional[go.Figure]:
    """
    Create a line chart for Jupiter perpetuals borrow rates data with leverage adjustment

    Args:
        data (Dict): API response data
        token_symbol (str): Token symbol for chart title
        leverage (float): Leverage multiplier

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

    # Convert hourly borrow rate to APY and apply leverage
    # avgHourlyBorrowRate is already in percentage, convert to APY by multiplying with 365*24
    df['hourly_borrow_rate_apy'] = df['avgHourlyBorrowRate'] * 365 * 24
    df['leveraged_borrow_rate_apy'] = df['hourly_borrow_rate_apy'] * leverage

    # Create the line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['hourBucket'],
        y=df['leveraged_borrow_rate_apy'],
        mode='lines+markers',
        name=f'{token_symbol} Leveraged Borrow Rate ({leverage}x)',
        line=dict(color='#FF6B35', width=2),  # Orange
        marker=dict(size=4),
        hovertemplate='<b>%{text}</b><br>' +
                      'Time: %{x}<br>' +
                      'Leveraged APY: %{y:.2f}%<br>' +
                      '<extra></extra>',
        text=[f'{token_symbol} ({leverage}x)'] * len(df)
    ))

    # Add base rate for comparison
    fig.add_trace(go.Scatter(
        x=df['hourBucket'],
        y=df['hourly_borrow_rate_apy'],
        mode='lines',
        name=f'{token_symbol} Base Borrow Rate (1x)',
        line=dict(color='#1f77b4', width=1, dash='dash'),  # Blue dashed
        opacity=0.7,
        hovertemplate='<b>%{text}</b><br>' +
                      'Time: %{x}<br>' +
                      'Base APY: %{y:.2f}%<br>' +
                      '<extra></extra>',
        text=[f'{token_symbol} Base'] * len(df)
    ))

    # Update layout
    fig.update_layout(
        title=f'{token_symbol} Jupiter Perps Borrow Rates APY (Leverage: {leverage}x)',
        xaxis_title='Time',
        yaxis_title='Annualized Borrow Rate (%)',
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

    # Format y-axis to show percentage with 2 decimal places
    fig.update_yaxes(tickformat='.2f')

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
