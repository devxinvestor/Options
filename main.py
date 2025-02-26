import yfinance as yf
import numpy as np
from arch import arch_model
from scipy.interpolate import griddata

from data_loader import get_options_chain, parse_options_chain
from volatility_model import VolatilityModel
from pricing_model import PricingModel
from signal_generator import SignalGenerator

ticker = 'AAPL'

options_json = get_options_chain(ticker)
options_json

options = parse_options_chain(options_json)

stock_data = yf.download(ticker, period="5y")

# get the interest rate from an api 10 year treasury yield
interest_rate = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1]

print("Options Chain:")
print(options.head())

stock_data['LogReturn'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
stock_data.dropna(inplace=True)

stock_data['LogReturn'] *= 100

garch_model = arch_model(stock_data['LogReturn'], vol='Garch', p=1, q=1)
garch_model_fit = garch_model.fit(disp='off')

print(garch_model_fit.summary())

underlying_price = stock_data['Close'].iloc[-1]
risk_free_rate = interest_rate / 100

volModel = VolatilityModel(options_chain_df=options, underlying_price=underlying_price, risk_free_rate=risk_free_rate)
historical_volatility = garch_model_fit.conditional_volatility

options['ImpliedVolatility'] = options.apply(
    lambda row: volModel.calculate_implied_volatility(
        option_price=row['mark'],
        strike=row['strikePrice'],
        expiration_days=row['daysToExpiration'],
        option_type=row['putCall'],  # 'CALL' or 'PUT'
    ),
    axis=1
)

# Display the first few rows of the options chain with implied volatilities
print("Options Chain with Implied Volatilities:")
print(options.head())

# Plot implied volatility vs strike price

strikes = options['strikePrice'].values
expirations = options['daysToExpiration'].values
implied_vols = options['ImpliedVolatility'].values

grid_strikes, grid_expirations = np.meshgrid(
    np.linspace(strikes.min(), strikes.max(), 100),
    np.linspace(expirations.min(), expirations.max(), 100)
)

# Add small noise to strikes and expirations
noise_level = 1e-5  # Adjust this value as needed
strikes = options['strikePrice'].values + np.random.normal(0, noise_level, size=len(options))
expirations = options['daysToExpiration'].values + np.random.normal(0, noise_level, size=len(options))
implied_vols = options['ImpliedVolatility'].values

# Create a grid for interpolation
grid_strikes, grid_expirations = np.meshgrid(
    np.linspace(strikes.min(), strikes.max(), 100),
    np.linspace(expirations.min(), expirations.max(), 100)
)

# Interpolate using cubic spline
volatility_surface = griddata(
    (strikes, expirations), implied_vols,
    (grid_strikes, grid_expirations), method='cubic'
)

underlying_price = stock_data['Close'].iloc[-1]
options['Moneyness'] = underlying_price[0] / options['strikePrice']
options['TimeToExpiration'] = options['daysToExpiration'] / 365
options['RiskFreeRate'] = interest_rate / 100
options['HistoricalVolatility'] = garch_model_fit.conditional_volatility[-1] / 100

print("Options Chain with Features:")
print(options.head())

pricing_model = PricingModel()
pricing_model.train_xgboost_model(options, underlying_price[0], interest_rate/100, historical_volatility[0])

predicted_chain = pricing_model.predict_option_chain(
    stock_price=stock_data['Close'].iloc[-1],
    options_chain=options,
    interest_rate=interest_rate,
    historical_volatility=historical_volatility.iloc[-1]
)

print("Predicted Option Chain:")
print(predicted_chain.head())

signal_generator = SignalGenerator()
signals = signal_generator.generate_trading_signals(predicted_chain)

print("\nTrading Signals:")
print(signal_generator.format_output(signals))