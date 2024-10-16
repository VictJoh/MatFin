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
r = 0.0418  
strike_prices = [70, 90, 100, 110, 130]
times = [1/12, 3/12, 1/2]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, sigma in enumerate(sigmas):
    for t in times:
        option_prices = [BlackScholes(S0, K, t, r, sigma) for K in strike_prices]
        axs[i].plot(strike_prices, option_prices, label=f'T = {t*12:.0f} months')
    
    axs[i].set_title(f'Option Price (σ = {sigma})')
    axs[i].set_xlabel('Strike Price')
    axs[i].set_ylabel('Option Price')
    axs[i].legend()

plt.tight_layout()
plt.savefig('OptionPrices.png')  
plt.show()

"""Part B"""

def option_data(ticker, expiration_date):
    stock = yf.Ticker(ticker)
    options_chain = stock.option_chain(expiration_date)
    calls = options_chain.calls
    return calls

def implied_volatility(S0, K, T, r, market_price, tol=0.005):
    func = lambda sigma: BlackScholes(S0, K, T, r, sigma) - market_price
    f_low, f_high = func(0.01), func(10.0)
    if f_low * f_high > 0:
        return np.nan
    try:
        result = brentq(func, f_low, f_high, xtol=tol)
        return result
    except ValueError:
        return np.nan


def plot_volatility(ticker, day):
    day = datetime.datetime.strptime(day, '%Y-%m-%d')
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    expiration_date = None

    for date in options_dates:
        exp_date = datetime.datetime.strptime(date, '%Y-%m-%d')
        if exp_date > day + datetime.timedelta(days=7):
            expiration_date = date
            break

    if expiration_date is None:
        print("No expiration date found beyond 7 days from the given day.")
        return

    calls = option_data(ticker, expiration_date)

    stock_info = stock.history(start=day, end=day + datetime.timedelta(days=1))
    if stock_info.empty:
        print(f"No stock data available for {day.date()}.")
        return
    S0 = stock_info['Close'].iloc[-1]

    r = 0.0418  
    market_prices = calls['lastPrice'].values 
    strike_prices = calls['strike'].values

    expiration_date_dt = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
    T = (expiration_date_dt - day).days / 365.25
    T = max(T, 1e-6)

    implied_volatilities = []
    valid_strikes = []

    for price, K in zip(market_prices, strike_prices):
        if price <= 0:
            implied_volatilities.append(np.nan)
            valid_strikes.append(K)
            continue
        vol = implied_volatility(S0, K, T, r, price)
        implied_volatilities.append(vol)
        valid_strikes.append(K)

    plt.figure(figsize=(10, 6))
    plt.scatter(valid_strikes, implied_volatilities, color='blue', label='Implied Volatility')
    plt.plot(valid_strikes, implied_volatilities, color='blue')
    
    plt.axvline(S0, color='red', linestyle='--', label=f'Underlying Price (S0 = {S0:.2f})')
    
    plt.title(f'Implied Volatility vs Strike Price for {ticker} on {day.date()}')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True, alpha=0.6)

    safe_day = day.strftime('%Y-%m-%d')
    filename = f'ImpliedVolatility_{ticker}_{safe_day}.png'
    plt.savefig(filename)  
    plt.show()

ticker = 'SPY'  
day = '2024-08-05' 
plot_volatility(ticker, day)

