import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import fetch_jupiter_perps_data, plot_jupiter_perps_borrow_rates_chart

st.set_page_config(
    page_title="Jupiter Perps Borrow Rates",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Jupiter Perps Borrow Rates")
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
                    import pandas as pd
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
