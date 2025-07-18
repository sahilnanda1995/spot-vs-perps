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

    # Convert APY to percentage
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

st.subheader("Spot")
st.write("Spot is a contract for difference (CFD) that allows you to trade the underlying asset directly.")

st.subheader("Perps")
st.write("Perps are a type of perpetual futures contract that allows you to trade the underlying asset directly.")
