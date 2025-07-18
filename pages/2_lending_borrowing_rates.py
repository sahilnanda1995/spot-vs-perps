import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_token_config, fetch_rates_data, plot_lending_borrowing_rates_chart

st.set_page_config(
    page_title="Lending & Borrowing Rates",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Lending & Borrowing Rates")
st.write("Select a token and protocol to view historical lending and borrowing rates.")

# Load token configuration
token_config = load_token_config()

if token_config:
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
                                import pandas as pd
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
else:
    st.error("Failed to load token configuration. Please check your token_config.json file.")
