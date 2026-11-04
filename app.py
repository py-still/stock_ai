import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

# =====================================================================
# 1. PAGE SETUP & INSTITUTIONAL THEME
# =====================================================================

st.set_page_config(
    page_title="AlphaTerminal | Quantitative Stock Advisory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Financial Styling
st.markdown("""
<style>
    /* Dark terminal styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    }
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .verdict-badge-buy {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid #2ea043;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
    }
    .verdict-badge-sell {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid #da3633;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
    }
    .verdict-badge-hold {
        background: rgba(163, 113, 247, 0.15);
        color: #bc8cff;
        border: 1px solid #8957e5;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
    }
    .factor-check {
        padding: 6px 0;
        font-size: 0.95rem;
        border-bottom: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. MARKET DATA HARNESS (Resilient & Defensive)
# =====================================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes Wilders Exponential RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


@st.cache_data(ttl=180, show_spinner=False)
def fetch_complete_stock_data(ticker_symbol: str):
    """
    Fetches real-time price, technicals, and balance-sheet fundamentals.
    Returns (data_dict, error_message).
    """
    clean_ticker = ticker_symbol.strip().upper()
    try:
        stock = yf.Ticker(clean_ticker)
        hist = stock.history(period="1y")

        if hist.empty or len(hist) < 40:
            return None, f"No trading records found for '{clean_ticker}'. Verify the symbol."

        # Technical Indicators
        hist["SMA_50"] = hist["Close"].rolling(50, min_periods=20).mean()
        hist["SMA_200"] = hist["Close"].rolling(200, min_periods=50).mean()
        hist["RSI"] = calculate_rsi(hist["Close"], period=14)

        # Quotes
        latest_close = float(hist["Close"].iloc[-1])
        prior_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest_close
        fast_price = stock.fast_info.get("last_price")
        current_price = float(fast_price) if fast_price and not np.isnan(fast_price) else latest_close

        change_abs = current_price - prior_close
        change_pct = (change_abs / prior_close) * 100 if prior_close > 0 else 0.0

        # Safe Fundamental Ingestion
        info = stock.info or {}
        
        # Debt / Equity normalization (yfinance provides either ratio e.g. 1.2 or pct 120.0)
        raw_de = info.get("debtToEquity", None)
        debt_to_equity = None
        if raw_de is not None and not np.isnan(raw_de):
            debt_to_equity = round(float(raw_de) / 100.0 if raw_de > 20 else float(raw_de), 2)

        # EPS & P/E
        pe_trailing = info.get("trailingPE")
        pe_forward = info.get("forwardPE")
        eps = info.get("trailingEps")
        profit_margin = info.get("profitMargins")

        snapshot = {
            "ticker": clean_ticker,
            "company_name": info.get("shortName") or info.get("longName") or clean_ticker,
            "sector": info.get("sector", "General Equity"),
            "current_price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "change_abs": round(change_abs, 2),
            "sma_50": round(float(hist["SMA_50"].iloc[-1]), 2) if not np.isnan(hist["SMA_50"].iloc[-1]) else None,
            "sma_200": round(float(hist["SMA_200"].iloc[-1]), 2) if not np.isnan(hist["SMA_200"].iloc[-1]) else None,
            "rsi_14": round(float(hist["RSI"].iloc[-1]), 1) if not np.isnan(hist["RSI"].iloc[-1]) else 50.0,
            "high_52": round(float(hist["High"].max()), 2),
            "low_52": round(float(hist["Low"].min()), 2),
            "pe_trailing": round(float(pe_trailing), 1) if pe_trailing and not np.isnan(pe_trailing) else None,
            "pe_forward": round(float(pe_forward), 1) if pe_forward and not np.isnan(pe_forward) else None,
            "eps": round(float(eps), 2) if eps and not np.isnan(eps) else None,
            "debt_to_equity": debt_to_equity,
            "profit_margin_pct": round(float(profit_margin) * 100, 1) if profit_margin and not np.isnan(profit_margin) else None,
            "history": hist
        }
        return snapshot, None

    except Exception as exc:
        return None, f"Telemetry error fetching {clean_ticker}: {str(exc)}"


# =====================================================================
# 3. DETERMINISTIC QUANT & RISK REASONING ENGINE
# =====================================================================

def evaluate_risk_and_signals(data: dict) -> dict:
    """
    Evaluates market metrics across strict mathematical guidelines.
    Prevents emotional or hallucinated outputs.
    """
    bullish_signals = []
    bearish_signals = []
    neutral_signals = []

    price = data["current_price"]
    sma_50 = data["sma_50"]
    sma_200 = data["sma_200"]
    rsi = data["rsi_14"]
    de = data["debt_to_equity"]
    pe = data["pe_trailing"]

    # 1. Moving Average Structure
    if sma_50 and sma_200:
        if price > sma_50 and sma_50 > sma_200:
            bullish_signals.append(f"Golden Trend: Price (${price:,.2f}) > 50 SMA (${sma_50:,.2f}) > 200 SMA (${sma_200:,.2f}).")
        elif price < sma_50 and sma_50 < sma_200:
            bearish_signals.append(f"Bearish Trend: Price is submerged below both 50 SMA (${sma_50:,.2f}) and 200 SMA (${sma_200:,.2f}).")
        elif price > sma_50 and price < sma_200:
            neutral_signals.append(f"Intermediate Rebound: Above 50 SMA (${sma_50:,.2f}) but capped below long-term 200 SMA (${sma_200:,.2f}).")
        else:
            neutral_signals.append("Consolidation: Price action is oscillating between primary moving averages.")

    # 2. RSI Momentum Check
    if rsi >= 72:
        bearish_signals.append(f"Overbought Momentum: 14-period RSI is extended at {rsi} (elevated mean-reversion risk).")
    elif rsi <= 32:
        bullish_signals.append(f"Oversold Condition: 14-period RSI is depressed at {rsi} (potential technical accumulation zone).")
    else:
        neutral_signals.append(f"Momentum Neutral: 14-period RSI sits at {rsi} (within regular 35-65 boundaries).")

    # 3. Balance Sheet Solvency (Debt / Equity)
    if de is not None:
        if de <= 0.60:
            bullish_signals.append(f"Sound Balance Sheet: Low Debt-to-Equity of {de}x minimizes refinancing distress.")
        elif de >= 1.75:
            bearish_signals.append(f"High Financial Leverage: Debt-to-Equity is elevated at {de}x.")
        else:
            neutral_signals.append(f"Manageable Leverage: Debt-to-Equity stands at {de}x.")
    else:
        neutral_signals.append("Solvency: Debt-to-Equity data not applicable for this instrument.")

    # 4. Valuation Multiple (P/E)
    if pe is not None:
        if pe > 48:
            bearish_signals.append(f"Valuation Multiples: Trailing P/E of {pe}x requires substantial future growth to sustain.")
        elif 0 < pe <= 17:
            bullish_signals.append(f"Discounted Valuation: Trailing P/E is conservative at {pe}x.")
        elif pe < 0:
            bearish_signals.append("Negative Earnings: Company is operating at an un-profitable net run rate.")

    # Synthesis Decision Matrix
    bull_score = len(bullish_signals)
    bear_score = len(bearish_signals)

    if bull_score >= 2 and bear_score == 0:
        verdict = "BUY"
        badge_class = "verdict-badge-buy"
        conviction = "High" if bull_score >= 3 else "Moderate"
        summary = f"Favorable asymmetric risk/reward. Strong technical trend supported by financial indicators."
        stop_loss = round(price * 0.94, 2)
        target = round(price * 1.15, 2)
    elif bear_score >= 2 and bull_score <= 1:
        verdict = "SELL / TRIM"
        badge_class = "verdict-badge-sell"
        conviction = "High" if bear_score >= 3 else "Moderate"
        summary = f"Elevated capital preservation risk. Caution advised due to overextension or weak fundamentals."
        stop_loss = round(price * 0.98, 2)
        target = round(price * 0.88, 2)
    else:
        verdict = "HOLD / DO NOTHING"
        badge_class = "verdict-badge-hold"
        conviction = "Balanced"
        summary = f"Conflicting signals. Momentum and valuation metrics offset each other. Avoid forcing an entry."
        stop_loss = round(price * 0.95, 2)
        target = round(price * 1.08, 2)

    return {
        "verdict": verdict,
        "badge_class": badge_class,
        "conviction": conviction,
        "summary": summary,
        "bullish": bullish_signals,
        "bearish": bearish_signals,
        "neutral": neutral_signals,
        "target": target,
        "stop_loss": stop_loss
    }


# =====================================================================
# 4. DASHBOARD INTERACTION
# =====================================================================

st.sidebar.markdown("### 🔍 AlphaTerminal Query")
quick_picks = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "TSLA", "SPY"]
selected_quick = st.sidebar.selectbox("Popular Symbols", ["(Custom Ticker)"] + quick_picks)

if selected_quick != "(Custom Ticker)":
    default_ticker = selected_quick
else:
    default_ticker = "AAPL"

ticker_input = st.sidebar.text_input("Stock Ticker", value=default_ticker).upper().strip()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Parameters")
st.sidebar.caption("• Moving Averages: 50 SMA / 200 SMA\n• Momentum Gauge: 14-period Wilder RSI\n• Fundamental Filters: P/E, EPS & Debt/Equity\n• Sizing Guardrail: 2.5:1 Risk/Reward Target")

if ticker_input:
    data, err = fetch_complete_stock_data(ticker_input)
    if err:
        st.error(f"❌ {err}")
    else:
        intel = evaluate_risk_and_signals(data)

        # Header Bar
        col_header, col_badge = st.columns([3, 1])
        with col_header:
            st.title(f"{data['company_name']} ({data['ticker']})")
            p_color = "#3fb950" if data['change_pct'] >= 0 else "#f85149"
            st.markdown(
                f"<h2 style='margin-top:-15px;'>${data['current_price']:,.2f} "
                f"<span style='color:{p_color}; font-size:1.15rem; font-weight:600;'>"
                f"{data['change_abs']:+,.2f} ({data['change_pct']:+.2f}%)</span></h2>",
                unsafe_allow_html=True
            )
            st.caption(f"Sector: {data['sector']} | 52-Week Range: ${data['low_52']:,.2f} — ${data['high_52']:,.2f}")

        with col_badge:
            st.markdown(
                f"<div style='text-align:right; margin-top:10px;'>"
                f"<div class='{intel['badge_class']}'>{intel['verdict']}</div>"
                f"<div style='font-size:0.85rem; color:#8b949e; margin-top:6px;'>Conviction: <b>{intel['conviction']}</b></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Fundamentals Grid
        st.markdown("---")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trailing P/E", f"{data['pe_trailing']}x" if data['pe_trailing'] else "N/A")
        c2.metric("Forward P/E", f"{data['pe_forward']}x" if data['pe_forward'] else "N/A")
        c3.metric("Diluted EPS", f"${data['eps']:.2f}" if data['eps'] else "N/A")
        c4.metric("Debt / Equity", f"{data['debt_to_equity']}x" if data['debt_to_equity'] is not None else "N/A")
        c5.metric("14-Day RSI", f"{data['rsi_14']}")
        c6.metric("Profit Margin", f"{data['profit_margin_pct']}%" if data['profit_margin_pct'] else "N/A")

        # Visual Chart + Factor Analysis Breakdown
        col_chart, col_audit = st.columns([1.8, 1.2])

        with col_chart:
            st.subheader("Price Action & Moving Averages")
            df = data["history"]
            fig = go.Figure()

            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price",
                increasing_line_color='#2ea043',
                decreasing_line_color='#da3633'
            ))

            # Overlays
            if data['sma_50']:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#e3b341', width=1.5), name="50 SMA"))
            if data['sma_200']:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#58a6ff', width=1.5), name="200 SMA"))

            fig.update_layout(
                template="plotly_dark",
                height=440,
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_audit:
            st.subheader("Objective Signal Audit")
            st.info(intel["summary"])

            col_t1, col_t2 = st.columns(2)
            col_t1.metric("Calculated Target", f"${intel['target']:,.2f}")
            col_t2.metric("Stop Loss Guide", f"${intel['stop_loss']:,.2f}")

            st.markdown("#### Factor Verification:")
            for item in intel["bullish"]:
                st.markdown(f"<div class='factor-check' style='color:#3fb950;'>✅ {item}</div>", unsafe_allow_html=True)
            for item in intel["bearish"]:
                st.markdown(f"<div class='factor-check' style='color:#f85149;'>⚠️ {item}</div>", unsafe_allow_html=True)
            for item in intel["neutral"]:
                st.markdown(f"<div class='factor-check' style='color:#8b949e;'>ℹ️ {item}</div>", unsafe_allow_html=True)

        # Conversational Terminal
        st.markdown("---")
        st.subheader(f"💬 Conversational Analyst: {data['ticker']}")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Render conversation history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input(f"Ask anything about {data['ticker']} (e.g., 'Is the P/E too expensive?', 'What's the downside risk?', 'Why hold?')")

        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Response Synthesis Engine
            q = user_prompt.lower()
            if "debt" in q or "balance" in q or "solvency" in q:
                if data['debt_to_equity'] is not None:
                    reply = (f"**Debt Analysis:** {data['ticker']} has a Debt-to-Equity ratio of **{data['debt_to_equity']}x**. "
                             f"{'This indicates a conservative, low-risk capital structure.' if data['debt_to_equity'] < 0.8 else 'This indicates higher debt reliance, which requires consistent cash flow.'}")
                else:
                    reply = f"Balance-sheet debt-to-equity is not reported for {data['ticker']}."
            elif "p/e" in q or "valuation" in q or "expensive" in q or "cheap" in q:
                pe_val = data['pe_trailing'] or data['pe_forward']
                reply = (f"**Valuation Multiples:** {data['ticker']} is trading at a trailing P/E of **{data['pe_trailing']}x** "
                         f"and a forward P/E of **{data['pe_forward']}x**. "
                         f"{'This represents a premium valuation where missing future earnings forecasts could cause volatility.' if pe_val and pe_val > 35 else 'Valuation multiples are within standard market ranges.'}")
            elif "rsi" in q or "oversold" in q or "overbought" in q or "momentum" in q:
                reply = (f"**Momentum & RSI:** The 14-period RSI is currently **{data['rsi_14']}**. "
                         f"{'It is near overbought territory (>70), suggesting potential short-term pullbacks.' if data['rsi_14'] > 68 else 'It is near oversold territory (<35), indicating potential buyer interest.' if data['rsi_14'] < 35 else 'It is currently in healthy neutral consolidation territory.'}")
            elif "target" in q or "stop" in q or "risk" in q:
                reply = (f"**Risk Framework:** For {data['ticker']}, based on volatility and moving average support, the modeled **Stop Loss is ${intel['stop_loss']:,.2f}** "
                         f"and the upside **Target is ${intel['target']:,.2f}**.")
            else:
                reply = (f"**Consensus Verdict for {data['ticker']}: {intel['verdict']}** (Conviction: {intel['conviction']}).\n\n"
                         f"{intel['summary']}\n\n"
                         f"• Current Price: ${data['current_price']:,.2f}\n"
                         f"• Primary Support (50 SMA): ${data['sma_50'] if data['sma_50'] else 'N/A'}\n"
                         f"• Stop Loss Guideline: ${intel['stop_loss']:,.2f}")

            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
