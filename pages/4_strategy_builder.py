import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_token_config, fetch_strategy_data, calculate_net_apy, plot_strategy_chart

st.set_page_config(
    page_title="Strategy Builder",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Strategy Builder")
st.write("Create long/short strategies with leverage to analyze potential returns.")

# Load token configuration
token_config = load_token_config()

if token_config:
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
else:
    st.error("Failed to load token configuration. Please check your token_config.json file.")
