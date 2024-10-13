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
    
    def calculate_efficient_frontier(self, num_portofolios = 100):
        mean_returns, _ = self.calculate_expected_returns_and_volatility()
        covariance_matrix = self.calculate_covariance_matrix()
        num_assets = len(mean_returns)

    
        target_returns = np.linspace(min(mean_returns), max(mean_returns), num_portofolios)

        frontier_volatilities = []
        frontier_returns = []

        for r in target_returns:
            w = self.find_min_var(r)
            weighted_r = self.weighted_return(w)
            weighted_var = self.weighted_variance(w, self.calculate_covariance_matrix())

            frontier_returns.append(weighted_r)
            frontier_volatilities.append(weighted_var)
            
        return frontier_returns, frontier_volatilities
    
    def plot_efficient_frontier(self):
            returns, volatilities = self.calculate_efficient_frontier()
            plt.figure(figsize=(10, 6))
            plt.plot(volatilities, returns, label='Efficient Frontier', color='blue')

            # Add titles and labels
            plt.title("Efficient Frontier", fontsize=16)
            plt.xlabel("Volatility (Standard Deviation)", fontsize=12)
            plt.ylabel("Expected Return", fontsize=12)

            # Add a legend
            plt.legend()

            # Show the plot
            plt.grid(True)
            plt.show()

    
    def plot_return_densities(self, ax, label_prefix):
        returns = self.calculate_returns()
        expected_returns, volatilities = self.calculate_expected_returns_and_volatility()

        asset_names = self.data['Adj Close'].columns
        limit = max(abs(returns.min().min()), abs(returns.max().max()))

        for i in range(len(asset_names)):
            asset_returns = returns[asset_names[i]]
            mu, std = expected_returns[i], volatilities[i]  
            ax[i].hist(asset_returns, bins=30, density = True)
            x = np.linspace(-limit, limit, 100)
            ax[i].plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
            ax[i].set_xlim(-limit, limit)
            ax[i].set_title(f'{label_prefix} Returns for {asset_names[i]}', fontsize=8)
            ax[i].set_ylabel('Density', fontsize=8)
            ax[i].tick_params(axis='x', labelsize=8)
            ax[i].tick_params(axis='y', labelsize=8)

daily = StockAnalysis('G:/My Drive/UiO/Semester3/MatFin/daily_stock_data.csv')
weekly = StockAnalysis('G:/My Drive/UiO/Semester3/MatFin/weekly_stock_data.csv')

cov_matrix = daily.calculate_covariance_matrix()
print("Covariance Matrix:")
print(cov_matrix)

corr_matrix = daily.calculate_correlation_matrix()
print("Correlation Matrix:")
print(corr_matrix)

max_assets = max(len(daily.data['Adj Close'].columns), len(weekly.data['Adj Close'].columns))

fig, axs = plt.subplots(max_assets, 2, figsize=(16, max_assets * 5))

daily.plot_return_densities(axs[:, 0], 'Daily')
weekly.plot_return_densities(axs[:, 1], 'Weekly')

plt.tight_layout()
plt.show()

daily.plot_efficient_frontier()

