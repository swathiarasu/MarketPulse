import yfinance as yf
import pandas as pd

def fetch_price_data(tickers, start, end):
    # Download full data, turn off auto-adjust to get 'Adj Close'
    raw_data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)

    # Try to extract Adjusted Close
    adj_close = pd.DataFrame()
    for ticker in tickers:
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Multiple tickers: Access via multi-index
                adj_close[ticker] = raw_data[ticker]['Adj Close']
            else:
                # Single ticker: flat columns
                adj_close[ticker] = raw_data['Adj Close']
        except KeyError:
            # Fallback to Close if Adj Close is missing
            print(f"[!] 'Adj Close' not found for {ticker}, using 'Close' instead.")
            if isinstance(raw_data.columns, pd.MultiIndex):
                adj_close[ticker] = raw_data[ticker]['Close']
            else:
                adj_close[ticker] = raw_data['Close']
    
    return adj_close.dropna()

# Example usage
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']
start_date = '2020-01-01'
end_date = '2024-12-31'

price_df = fetch_price_data(tickers, start_date, end_date)
print(price_df.head())



# Fill missing values using forward fill
cleaned_price_df = price_df.ffill().dropna()

import numpy as np

# Calculate daily log returns
def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

returns_df = calculate_log_returns(cleaned_price_df)
returns_df.head()

# Correlation Matrix
correlation_matrix = returns_df.corr()

# Covariance Matrix
covariance_matrix = returns_df.cov()

print("Correlation Matrix:\n", correlation_matrix)
print("Covariance Matrix:\n", covariance_matrix)

#####################################################################

# Define your weights (must sum to 1)
portfolio_weights = {
    'AAPL': 0.25,
    'MSFT': 0.25,
    'GOOGL': 0.20,
    'AMZN': 0.20,
    'SPY': 0.10  # This can be a hedge or part of the portfolio
}

# Convert to numpy array in order of columns
weights_array = np.array([portfolio_weights[ticker] for ticker in returns_df.columns])

# Daily portfolio returns
portfolio_returns = returns_df.dot(weights_array)

# Annualized Volatility = std deviation * sqrt(252 trading days)
annual_volatility = portfolio_returns.std() * np.sqrt(252)
print(f"Annualized Portfolio Volatility: {annual_volatility:.2%}")


# Ensure benchmark returns are available
benchmark_returns = returns_df['SPY']
portfolio_excl_spy = returns_df.drop(columns='SPY')  # Portfolio excluding SPY

# Recalculate weights accordingly
weights_excl_spy = np.array([portfolio_weights[ticker] for ticker in portfolio_excl_spy.columns])
portfolio_core_returns = portfolio_excl_spy.dot(weights_excl_spy)

# Beta = Cov(portfolio, market) / Var(market)
cov_matrix = np.cov(portfolio_core_returns, benchmark_returns)
beta = cov_matrix[0, 1] / cov_matrix[1, 1]
print(f"Portfolio Beta vs SPY: {beta:.3f}")

risk_free_rate_daily = 0.04 / 252
excess_returns = portfolio_returns - risk_free_rate_daily

# Sharpe Ratio = (mean excess return / std deviation) * sqrt(252)
sharpe_ratio = (excess_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Asset Correlation Matrix")
plt.show()

#################################################################################
confidence_level = 0.95
holding_period = 1  # in days
initial_investment = 1_000_000  # $1M portfolio

def historical_var(returns, confidence, investment):
    var_percentile = np.percentile(returns, (1 - confidence) * 100)
    return -var_percentile * investment

hist_var = historical_var(portfolio_returns, confidence_level, initial_investment)
print(f"Historical VaR (95%): ${hist_var:,.2f}")

from scipy.stats import norm

def parametric_var(mean, std_dev, confidence, investment, horizon=1):
    z_score = norm.ppf(1 - confidence)
    var = -(mean * horizon + z_score * std_dev * np.sqrt(horizon))
    return var * investment

param_var = parametric_var(portfolio_returns.mean(), portfolio_returns.std(), confidence_level, initial_investment)
print(f"Parametric VaR (95%): ${param_var:,.2f}")

def monte_carlo_var(mean, std_dev, investment, confidence, num_simulations=10000, horizon=1):
    simulated_returns = np.random.normal(mean, std_dev, num_simulations)
    simulated_portfolio_values = investment * (1 + simulated_returns)
    simulated_losses = investment - simulated_portfolio_values
    var = np.percentile(simulated_losses, 100 * (1 - confidence))
    return var

mc_var = monte_carlo_var(portfolio_returns.mean(), portfolio_returns.std(), initial_investment, confidence_level)
print(f"Monte Carlo VaR (95%): ${mc_var:,.2f}")

print("\n--- Value at Risk Summary ---")
print(f"Historical VaR (95%):       ${hist_var:,.2f}")
print(f"Parametric VaR (95%):       ${param_var:,.2f}")
print(f"Monte Carlo VaR (95%):      ${mc_var:,.2f}")

n_day = 5
print(f"5-Day Parametric VaR (95%): ${param_var * np.sqrt(n_day):,.2f}")


################################################################################


# Define crisis windows
crisis_periods = {
    '2008 Financial Crisis': ('2008-09-01', '2009-03-01'),
    'COVID-19 Crash': ('2020-02-15', '2020-04-15')
}
def simulate_crisis_impact(returns_df, weights, crisis_periods, initial_investment):
    results = {}

    weights_array = np.array([weights[t] for t in returns_df.columns])
    
    for name, (start, end) in crisis_periods.items():
        try:
            crisis_returns = returns_df.loc[start:end]
            portfolio_returns = crisis_returns.dot(weights_array)
            cumulative_return = (1 + portfolio_returns).prod() - 1
            value_change = initial_investment * cumulative_return
            results[name] = {
                'Total Return (%)': round(cumulative_return * 100, 2),
                'Value Lost ($)': round(initial_investment - (initial_investment + value_change), 2)
            }
        except Exception as e:
            results[name] = f"Error during simulation: {e}"

    return results

crisis_results = simulate_crisis_impact(returns_df, portfolio_weights, crisis_periods, initial_investment)

print("\n--- Historical Stress Test Results ---")
for scenario, result in crisis_results.items():
    print(f"\nScenario: {scenario}")
    for k, v in result.items():
        print(f"{k}: {v}")

def simulate_custom_crash(weights, affected_assets, drop_percent, initial_investment):
    simulated_loss = 0.0
    for asset, weight in weights.items():
        if asset in affected_assets:
            simulated_loss += weight * (drop_percent / 100)
    estimated_loss = simulated_loss * initial_investment
    return round(estimated_loss, 2)

tech_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
estimated_loss = simulate_custom_crash(portfolio_weights, tech_assets, -40, initial_investment)

print(f"\nHypothetical Tech Crash (-40%) Estimated Portfolio Loss: ${estimated_loss:,.2f}")

############################################################################################
risk_metrics = {
    'Volatility (Annualized)': f"{annual_volatility:.2%}",
    'Beta vs SPY': round(beta, 3),
    'Sharpe Ratio': round(sharpe_ratio, 2),
    'VaR (Historical 95%) [$]': round(hist_var, 2),
    'VaR (Parametric 95%) [$]': round(param_var, 2),
    'VaR (Monte Carlo 95%) [$]': round(mc_var, 2),
    'Hypothetical Tech Crash Loss [$]': estimated_loss
}

# Add stress test results to this dictionary
for name, result in crisis_results.items():
    risk_metrics[f'{name} - Return (%)'] = result['Total Return (%)']
    risk_metrics[f'{name} - Value Lost ($)'] = result['Value Lost ($)']

import pandas as pd

# Convert to DataFrame with one row
risk_df = pd.DataFrame([risk_metrics])
risk_df.to_csv("Data/risk_dashboard_summary.csv", index=False)
print("Exported to risk_dashboard_summary.csv")

risk_df.to_excel("Data/risk_dashboard_summary.xlsx", index=False)
print("Exported to risk_dashboard_summary.xlsx")

risk_df.to_json("Data/risk_dashboard_summary.json", orient='records', lines=True)
print("Exported to risk_dashboard_summary.json")

###########################################################################

import os

os.makedirs("data", exist_ok=True)

# Export portfolio-level daily returns
portfolio_returns.to_csv("data/portfolio_daily_returns.csv", header=["Portfolio Return"])

# Export individual asset returns
returns_df.to_csv("data/individual_asset_returns.csv")

import matplotlib.pyplot as plt

labels = list(crisis_results.keys()) + ["Hypothetical Tech Crash"]
losses = [res['Value Lost ($)'] for res in crisis_results.values()] + [estimated_loss]

plt.figure(figsize=(10, 6))
plt.bar(labels, losses, color='salmon')
plt.title("Stress Testing: Portfolio Value at Risk Under Extreme Scenarios")
plt.ylabel("Loss ($)")
plt.xticks(rotation=20)
plt.tight_layout()

# Save to file
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/stress_test_chart.png")
plt.close()
