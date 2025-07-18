import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Configuration constants for generic token pair system
STRATEGY_CONFIG = {
    'asgard_spot': {
        'token_a': 'JitoSOL',  # Long position token
        'token_b': 'USDC',     # Short position token
        'fee_structure': {
            'opening_rate': 0.001,  # 0.1%
            'closing_rate': 0.0     # 0%
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

def calculate_sol_fees(jupiter_perps_data: Dict, principal_amount: float, leverage: float) -> Optional[pd.DataFrame]:
    """
    Calculate fees for leveraged SOL trading on Jupiter Perpetuals

    Args:
        jupiter_perps_data (Dict): API response data for Jupiter Perps SOL
        principal_amount (float): The principal amount in USD
        leverage (float): The leverage multiplier

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

def calculate_token_pair_strategy_apy(market_data: Dict, leverage: float) -> Optional[pd.DataFrame]:
    """
    Calculate strategy APY for a token pair market

    Args:
        market_data (Dict): Market data including rates and staking
        leverage (float): Leverage multiplier

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
            metadata['token_b_symbol']
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
            strategy_df = calculate_token_pair_strategy_apy(market_data, leverage)

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

def plot_multi_strategy_comparison_chart(
    jupiter_data: Optional[pd.DataFrame],
    asgard_markets_fees: Dict,
    leverage: float
) -> Optional[go.Figure]:
    """
    Create a chart showing Jupiter Perps and all Asgard markets on the same hourly rate scale

    Args:
        jupiter_data (Optional[pd.DataFrame]): Jupiter Perps fee data
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

    # Add Asgard market traces
    color_index = 1
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
    asgard_count = len([k for k, v in asgard_markets_fees.items()
                      if v.get('strategy_df') is not None and not v['strategy_df'].empty])

    if jupiter_available and asgard_count > 0:
        title = 'Multi-Strategy Hourly Rates Comparison'
    elif jupiter_available:
        title = 'Jupiter Perps - Hourly Borrow Rates'
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

    # Jupiter Perps Borrow Rates Section
    st.divider()
    st.subheader("🚀 Jupiter Perps Borrow Rates")
    st.write("Select a token and leverage to view its historical borrow rates on Jupiter Perpetuals.")

    # Token and leverage selection
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Jupiter perps supports only SOL, ETH, and BTC
        jupiter_perps_tokens = ["SOL", "ETH", "BTC"]
        selected_jupiter_perps_token = st.selectbox(
            "Choose a token:",
            options=jupiter_perps_tokens,
            index=0,
            key="jupiter_perps_token_select"
        )

    with col2:
        leverage = st.number_input(
            "Leverage:",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="jupiter_perps_leverage"
        )

    with col3:
        jupiter_perps_time_period = st.selectbox(
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
            key="jupiter_perps_time_period_select"
        )

    if st.button("Fetch Jupiter Perps Data", type="primary", key="fetch_jupiter_perps_btn"):
        with st.spinner(f"Fetching {selected_jupiter_perps_token} Jupiter perps borrow rates data..."):
            # Fetch data
            jupiter_perps_data = fetch_jupiter_perps_data(selected_jupiter_perps_token, limit=jupiter_perps_time_period[1])

            if jupiter_perps_data:
                # Plot chart
                fig = plot_jupiter_perps_borrow_rates_chart(jupiter_perps_data, selected_jupiter_perps_token, leverage)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # Show some basic statistics
                    records = jupiter_perps_data.get('data', {}).get('records', [])
                    if records:
                        df = pd.DataFrame(records)
                        # Convert hourly rate to APY
                        df['hourly_borrow_rate_apy'] = df['avgHourlyBorrowRate'] * 365 * 24
                        df['leveraged_borrow_rate_apy'] = df['hourly_borrow_rate_apy'] * leverage

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            current_leveraged_apy = df['leveraged_borrow_rate_apy'].iloc[0]
                            st.metric("Current Leveraged APY", f"{current_leveraged_apy:.2f}%")

                        with col2:
                            avg_leveraged_apy = df['leveraged_borrow_rate_apy'].mean()
                            st.metric("Average Leveraged APY", f"{avg_leveraged_apy:.2f}%")

                        with col3:
                            min_leveraged_apy = df['leveraged_borrow_rate_apy'].min()
                            st.metric("Min Leveraged APY", f"{min_leveraged_apy:.2f}%")

                        with col4:
                            max_leveraged_apy = df['leveraged_borrow_rate_apy'].max()
                            st.metric("Max Leveraged APY", f"{max_leveraged_apy:.2f}%")

                        # Show base rate metrics
                        st.markdown("#### Base Rate Metrics (1x leverage)")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            current_base_apy = df['hourly_borrow_rate_apy'].iloc[0]
                            st.metric("Current Base APY", f"{current_base_apy:.2f}%")

                        with col2:
                            avg_base_apy = df['hourly_borrow_rate_apy'].mean()
                            st.metric("Average Base APY", f"{avg_base_apy:.2f}%")

                        with col3:
                            min_base_apy = df['hourly_borrow_rate_apy'].min()
                            st.metric("Min Base APY", f"{min_base_apy:.2f}%")

                        with col4:
                            max_base_apy = df['hourly_borrow_rate_apy'].max()
                            st.metric("Max Base APY", f"{max_base_apy:.2f}%")

                        # Show calculation explanation
                        with st.expander("💡 Calculation Details"):
                            st.markdown(f"""
                            **Borrow Rate Calculation:**

                            - **Hourly Rate**: The raw `avgHourlyBorrowRate` from Jupiter (already in %)
                            - **Base APY**: Hourly Rate × 365 × 24 = {df['avgHourlyBorrowRate'].iloc[0]:.6f}% × 8760 = {current_base_apy:.2f}%
                            - **Leveraged APY**: Base APY × Leverage = {current_base_apy:.2f}% × {leverage} = {current_leveraged_apy:.2f}%

                            The leveraged APY represents the total borrowing cost when using {leverage}x leverage.
                            """)

                        # Show raw data (optional)
                        with st.expander("View Raw Data"):
                            display_df = df[['hourBucket', 'avgHourlyBorrowRate', 'hourly_borrow_rate_apy', 'leveraged_borrow_rate_apy', 'sampleCount']].copy()
                            display_df.columns = ['Time', 'Hourly Rate (%)', 'Base APY (%)', f'Leveraged APY ({leverage}x) (%)', 'Sample Count']
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
            max_value=100000.0,
            value=1000.0,
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
                ("1 Month", 720),
                ("15 Days", 360),
                ("1 Week", 168),
                ("3 Days", 72),
                ("1 Day", 24),
                ("12 Hours", 12)
            ],
            format_func=lambda x: x[0],
            index=2,
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

    if st.button("Analyze SOL Fees", type="primary", key="analyze_sol_fees_btn"):
        with st.spinner("Fetching fee data and calculating costs for both positions..."):
            # Fetch Jupiter perps data for SOL
            jupiter_perps_data = fetch_jupiter_perps_data("SOL", limit=sol_fee_time_period[1])

            # Fetch Asgard strategy data
            asgard_data = fetch_all_asgard_markets_data(principal_amount, sol_leverage, limit=sol_fee_time_period[1])

            jupiter_success = jupiter_perps_data is not None
            asgard_success = asgard_data and len(asgard_data) > 0  # Check if we have any market data
            time_period_hours = sol_fee_time_period[1]  # Define this early for use in calculations

            if jupiter_success or asgard_success:
                jupiter_fees_calculated = False
                asgard_fees_calculated = False

                # Calculate Jupiter Perps fees if data available
                if jupiter_success:
                    sol_fees_df = calculate_sol_fees(jupiter_perps_data, principal_amount, sol_leverage)
                    if sol_fees_df is not None and not sol_fees_df.empty:
                        jupiter_fees_calculated = True

                # Calculate Asgard fees if data available
                if asgard_success:
                    asgard_fees = calculate_multi_market_fees(asgard_data, principal_amount, sol_leverage, sol_fee_time_period[1])
                    if asgard_fees:
                        asgard_fees_calculated = True

                if jupiter_fees_calculated or asgard_fees_calculated:
                    # Plot multi-strategy comparison chart
                    fig = plot_multi_strategy_comparison_chart(
                        sol_fees_df if jupiter_fees_calculated else None,
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

                        if asgard_fees_calculated:
                            for market_key, market_fees in asgard_fees.items():
                                if jupiter_fees_calculated:
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

**Key Differences:**
- **Jupiter Perps**: Direct hourly borrow rate (always positive cost)
- **Asgard**: Equivalent hourly rate from net APY (negative = profitable, positive = unprofitable)
- **Chart**: All lines on same scale for direct comparison. Asgard below 0% means it's profitable!
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

st.subheader("Spot")
st.write("Spot is a contract for difference (CFD) that allows you to trade the underlying asset directly.")

st.subheader("Perps")
st.write("Perps are a type of perpetual futures contract that allows you to trade the underlying asset directly.")
