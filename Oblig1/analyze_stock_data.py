import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

class StockAnalysis:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, header=[0, 1], index_col=0, parse_dates=True)

    def calculate_returns(self):
        adj_close_prices = self.data['Adj Close']
        returns = (adj_close_prices / adj_close_prices.shift(1)) - 1
        return returns

    def calculate_expected_returns_and_volatility(self):
        returns = self.calculate_returns()
        mean_returns = returns.mean(axis=0)
        volatility = returns.std(axis=0)
        return mean_returns, volatility

    def sharpe_ratio(self, risk_free_r):
        mean_returns, volatility = self.calculate_expected_returns_and_volatility()
        return (mean_returns - risk_free_r)/volatility
    
    def calculate_covariance_matrix(self):
        returns = self.calculate_returns()
        covariance_matrix = returns.cov()
        return covariance_matrix

    def calculate_correlation_matrix(self):
        returns = self.calculate_returns()
        correlation_matrix = returns.corr()
        return correlation_matrix

    def weighted_variance(self, w, covariance_matrix):
        return np.dot(np.dot(w.T, covariance_matrix),w)
    
    def weighted_return(self, w):
        expected_return, _ = self.calculate_expected_returns_and_volatility()
        return np.dot(w.T, expected_return)
         
    def find_min_var(self, target_return):
        mean_returns, _ = self.calculate_expected_returns_and_volatility()
        covariance_matrix = self.calculate_covariance_matrix()
        num_assets = len(mean_returns)

        linear_constraint = LinearConstraint(np.ones(num_assets), 1, 1)

        target_return_constraint = LinearConstraint(mean_returns, target_return, target_return)

        bounds = Bounds(0, 1)


        w_0 = num_assets * [1. / num_assets]

        result = minimize(self.weighted_variance, w_0, args=(covariance_matrix,), method='SLSQP', bounds=bounds, constraints=[linear_constraint, target_return_constraint])
        
        w = result.x
        return w
    
    def calculate_efficient_frontier(self, num_portofolios=100):
        mean_returns, _ = self.calculate_expected_returns_and_volatility()
        covariance_matrix = self.calculate_covariance_matrix()
        num_assets = len(mean_returns)

        target_returns = np.linspace(min(mean_returns) * 0.5, max(mean_returns) * 2, num_portofolios)

        frontier_volatilities = []
        frontier_returns = []

        for r in target_returns:
            w = self.find_min_var(r)
            weighted_r = self.weighted_return(w)
            weighted_var = self.weighted_variance(w, covariance_matrix)
            weighted_std = np.sqrt(weighted_var)

            frontier_returns.append(weighted_r)
            frontier_volatilities.append(weighted_std)
        return frontier_returns, frontier_volatilities
    
    def plot_efficient_frontier(self):
        returns, volatilities = self.calculate_efficient_frontier()
        plt.figure(figsize=(10, 6))
        plt.plot(volatilities, returns, label='Efficient Frontier', color='blue')
        
        expected_returns, individual_volatilities = self.calculate_expected_returns_and_volatility()
        
        plt.scatter(individual_volatilities, expected_returns, marker='o', color='red', label='Individual Stocks')
        
        asset_names = self.data['Adj Close'].columns
        for i, asset in enumerate(asset_names):
            plt.annotate(asset, 
                        (individual_volatilities[i], expected_returns[i]),
                        textcoords="offset points", 
                        xytext=(5,5),  
                        ha='left', 
                        fontsize=8,
                        color='black')
        

        plt.title("Efficient Frontier", fontsize=16)
        plt.xlabel("Volatility (Standard Deviation)", fontsize=12)
        plt.ylabel("Expected Return", fontsize=12)

        plt.legend()

        plt.grid(True)
        plt.savefig('EfficientFrontier.png')  
        plt.show()
    def find_global_min_var(self):
        covariance_matrix = self.calculate_covariance_matrix()
        num_assets = len(covariance_matrix)
        linear_constraint = LinearConstraint(np.ones(num_assets), 1, 1)

        bounds = Bounds(0, 1)

        w_0 = num_assets * [1. / num_assets]

        def portfolio_variance(w):
            return np.dot(w.T, np.dot(covariance_matrix, w))

        result = minimize(portfolio_variance, w_0, method='SLSQP', bounds=bounds, constraints=[linear_constraint])

        w = result.x
        min_variance = result.fun
        return w, min_variance
    
    def plot_return_densities(self, axs, label_prefix):
        returns = self.calculate_returns()
        expected_returns, volatilities = self.calculate_expected_returns_and_volatility()

        asset_names = self.data['Adj Close'].columns
        limit = max(abs(returns.min().min()), abs(returns.max().max()))

        for i, asset in enumerate(asset_names):
            asset_returns = returns[asset].dropna()
            mu, std = expected_returns[asset], volatilities[asset]
            axs[i].hist(asset_returns, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
            x = np.linspace(-limit, limit, 100)
            axs[i].plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
            axs[i].set_xlim(-limit, limit)
            axs[i].set_title(f'{label_prefix} Returns for {asset}', fontsize=10)
            axs[i].set_xlabel('Return', fontsize=8)
            axs[i].set_ylabel('Density', fontsize=8)
            axs[i].tick_params(axis='both', labelsize=8)
        plt.tight_layout()
        plt.savefig(f'{label_prefix}_Return_Densities.png')  

daily = StockAnalysis('daily_stock_data.csv')
weekly = StockAnalysis('weekly_stock_data.csv')

cov_matrix = daily.calculate_covariance_matrix()
print("Daily Covariance Matrix:")
print(cov_matrix)

corr_matrix = daily.calculate_correlation_matrix()
print("\nDaily Correlation Matrix:")
print(corr_matrix)

max_assets = max(len(daily.data['Adj Close'].columns), len(weekly.data['Adj Close'].columns))

fig_daily, axs_daily = plt.subplots(max_assets, 1, figsize=(10, max_assets * 3), constrained_layout=True)
if max_assets == 1:
    axs_daily = [axs_daily] 
daily.plot_return_densities(axs_daily, 'Daily')
plt.show()

fig_weekly, axs_weekly = plt.subplots(max_assets, 1, figsize=(10, max_assets * 3), constrained_layout=True)
if max_assets == 1:
    axs_weekly = [axs_weekly]  
weekly.plot_return_densities(axs_weekly, 'Weekly')
plt.show()

plt.tight_layout()
plt.show()


weekly.plot_efficient_frontier()

w, min_variance = weekly.find_global_min_var()

print("Global Minimum Variance Portfolio Weights:")
for asset, weight in zip(weekly.data['Adj Close'].columns, w):
    print(f"{asset}: {weight:.4f}")
min_port_return = weekly.weighted_return(w)
min_port_volatility = np.sqrt(min_variance)
print(f"\nMinimum Variance: {min_variance:.6f}")
print(f"Expected Return of Minimum Variance Portfolio: {min_port_return:.6f}")
print(f"Volatility (Std Dev) of Minimum Variance Portfolio: {min_port_volatility:.6f}")