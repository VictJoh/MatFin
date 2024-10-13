import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from scipy.optimize import brentq

def BlackScholes(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S0 * norm.cdf(d1) - K*np.exp(-r*T) * norm.cdf(d2)
    return C
"""Part A"""
S0 = 100
sigmas = [0.15, 0.30, 0.45]
r = 0.0418 # risk free interest rate picked as https://ycharts.com/indicators/1_year_treasury_rate
strike_prices = [70, 90, 100, 110, 130]
times = [1/12, 3/12, 1/2]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, sigma in enumerate(sigmas):
    for t in times:
        option_prices = [BlackScholes(S0, K, t, r, sigma) for K in strike_prices]
        axs[i].plot(strike_prices, option_prices, label=f'T = {t*12:.0f} months')
    
    axs[i].set_title(f'Option Price (sigma = {sigma})')
    axs[i].set_xlabel(f'Strike Price')
    axs[i].set_ylabel(f'Option Price')
    axs[i].legend()

plt.tight_layout()
plt.show()

"""Part B"""

def option_data(ticker):
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    expiration = options_dates[0]  
    options_chain = stock.option_chain(expiration)
    calls = options_chain.calls
    return calls, expiration

def implied_volatility(S0, K, T, r, market_price, tol = 0.005):
    func = lambda sigma: BlackScholes(S0, K, T, r, sigma) - market_price
    f_low, f_high = func(0.01), func(5.0)
    if f_low * f_high > 0:
        return np.nan
    result = brentq(func, 0.01, 5.0, xtol=tol)
    return result


def plot_volatility(ticker, day):
    day = datetime.datetime.strptime(day, '%Y-%m-%d')
    calls, expiration = option_data(ticker)  
    S0 = calls['lastPrice'].iloc[0]
    r = 0.0418

    market_prices = calls['lastPrice'].values 
    strike_prices = calls['strike'].values

    expiration_date = datetime.datetime.strptime(expiration, '%Y-%m-%d')
    T = (expiration_date - day).days / 365.25
    T = max(T, 1e-6)

    implied_volatilities = []
    valid_strikes = []

    for price, K in zip(market_prices, strike_prices):
        vol = implied_volatility(S0, K, T, r, price)
        implied_volatilities.append(vol)
        valid_strikes.append(K)
    

    plt.figure(figsize=(10, 6))
    plt.scatter(valid_strikes, implied_volatilities, color='blue', label='Implied Volatility')
    plt.plot(valid_strikes, implied_volatilities, color='blue')
    plt.title(f'Implied Volatility vs Strike Price for {ticker}')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.show()

ticker = 'NVDA'  
day = '2024-10-08' 
plot_volatility(ticker, day)