import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_token_config, fetch_staking_data, plot_staking_apy_chart

st.set_page_config(
    page_title="Staking APY Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Staking APY Analysis")
st.write("Select a token to view its historical staking APY data.")

# Load token configuration
token_config = load_token_config()

if token_config:
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
                ("Last 30 Days", 720),
                ("Last 15 Days", 360),
                ("Last 7 Days", 168),
                ("Last 5 Days", 120),
                ("Last 3 Days", 72),
                ("Last Day", 24),
                ("Last 12 Hours", 12)
            ],
            format_func=lambda x: x[0],
            index=3
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
                        import pandas as pd
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
else:
    st.error("Failed to load token configuration. Please check your token_config.json file.")
