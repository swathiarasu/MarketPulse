import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# ------------------------
# Sidebar Inputs
# ------------------------
with st.sidebar:
    st.header("Portfolio Settings")

    tickers = st.multiselect(
        "Select Tickers",
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'SPY'],
        default=['AAPL', 'MSFT', 'GOOGL']
    )

    st.markdown("### Portfolio Weights")
    weights = [st.slider(f"{ticker} Weight", 0.0, 1.0, 1.0 / len(tickers), step=0.01) for ticker in tickers]
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"Total weights = {total_weight:.2f}. They should sum to 1.")
    weights = np.array(weights) / sum(weights)

    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

    confidence = st.selectbox("VaR Confidence Level", [0.90, 0.95, 0.99], index=1)
    initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=1000000, step=10000)

# ------------------------
# Data Fetching & Returns
# ------------------------
@st.cache_data(ttl=86400)
def get_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna()

price_df = get_price_data(tickers, start_date, end_date)
returns_df = np.log(price_df / price_df.shift(1)).dropna()
portfolio_returns = returns_df.dot(weights)

# ------------------------
# Risk Metrics
# ------------------------
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
hist_var = -np.percentile(portfolio_returns, (1 - confidence) * 100) * initial_investment
z = norm.ppf(1 - confidence)
param_var = -(portfolio_returns.mean() + z * portfolio_returns.std()) * initial_investment
simulated = np.random.normal(portfolio_returns.mean(), portfolio_returns.std(), 10000)
mc_losses = initial_investment - initial_investment * (1 + simulated)
mc_var = np.percentile(mc_losses, 100 * (1 - confidence))

# Expected Shortfall
cvar = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, (1 - confidence) * 100)].mean() * initial_investment

# Max Drawdown
cumulative = (1 + portfolio_returns).cumprod()
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak
max_drawdown = drawdown.min()

# ------------------------
# Display Metrics
# ------------------------
st.title("Interactive Risk Management Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Volatility (Annual)", f"{volatility:.2%}")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Max Drawdown", f"{max_drawdown:.2%}")

col1.metric(f"VaR (Historical {int(confidence*100)}%)", f"${hist_var:,.2f}")
col2.metric("VaR (Parametric)", f"${param_var:,.2f}")
col3.metric("VaR (Monte Carlo)", f"${mc_var:,.2f}")

st.metric("Expected Shortfall (CVaR)", f"${-cvar:,.2f}")

# ------------------------
# Return Distribution Plot
# ------------------------
st.subheader("Return Distribution")
fig1 = px.histogram(portfolio_returns, nbins=50, title="Daily Log Returns Distribution")
st.plotly_chart(fig1, use_container_width=True)

# ------------------------
# Correlation Heatmap
# ------------------------
st.subheader("Asset Correlation Heatmap")
fig2, ax = plt.subplots()
sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig2)

# ------------------------
# Cumulative Return Plot
# ------------------------
st.subheader("Portfolio Value Over Time")
st.line_chart(cumulative * initial_investment)

# ------------------------
# Allocation Pie Chart
# ------------------------
st.subheader("Portfolio Allocation")
st.plotly_chart(px.pie(values=weights, names=tickers, title="Asset Allocation"))

# ------------------------
# Stress Testing Module
# ------------------------
st.header("Stress Testing (Hypothetical Crash)")
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
default_tech = [t for t in tickers if t in tech_stocks]
crash_tickers = st.multiselect("Select assets to simulate crash on", tickers, default=default_tech)
crash_percent = st.slider("Crash severity (%)", min_value=10, max_value=90, value=40, step=5)

if st.button("Run Stress Test"):
    stress_loss = sum([w for t, w in zip(tickers, weights) if t in crash_tickers]) * (crash_percent / 100)
    estimated_loss = stress_loss * initial_investment
    st.subheader("Estimated Portfolio Loss")
    st.error(f" ${estimated_loss:,.2f}")
    st.plotly_chart(px.bar(x=["Stress Loss"], y=[-estimated_loss], labels={'x': "Scenario", 'y': "Loss ($)"}, title="Stress Test Impact"))

# ------------------------
# Educational Layer
# ------------------------
with st.expander(" Glossary"):
    st.markdown("""
**Volatility**: How much your investment returns move up and down.  
**Sharpe Ratio**: How much return you get for each unit of risk.  
**VaR (Value at Risk)**: The worst loss you might expect with a certain confidence level.  
**CVaR (Expected Shortfall)**: The average loss when things go really bad.  
**Drawdown**: The biggest fall from a peak in your portfolio.  
**Stress Test**: A simulation of what might happen if certain assets crash.
""")
