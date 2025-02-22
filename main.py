import yfinance as yf
import numpy as np
from data_loader import options_chain
from volatility_model import VolatilityModel
from pricing_model import PricingModel
from signal_generator import SignalGenerator

def main():
    # User input
    ticker = input("Enter the ticker symbol: ")

    # Fetch options chain and stock data
    options, interest_rate, implied_volatility = options_chain(ticker)
    stock_data = yf.download(ticker, period="5y")

    # Calculate historical volatility using VolatilityModel
    volatility_model = VolatilityModel()
    stock_data['LogReturn'] = volatility_model.calculate_log_returns(stock_data['Adj Close'])
    historical_volatility = volatility_model.calculate_garch_volatility(stock_data['Adj Close'])

    # Prepare features and labels for training
    options['Moneyness'] = stock_data['Adj Close'].iloc[-1] / options['strikePrice']
    options['TimeToExpiration'] = options['daysToExpiration'] / 365
    options['RiskFreeRate'] = interest_rate / 100
    options['HistoricalVolatility'] = historical_volatility.iloc[-1]

    # Interpolate the volatility surface
    options = volatility_model.interpolate_vol_surface(options)

    # Train the XGBoost model
    pricing_model = PricingModel()
    pricing_model.train_xgboost_model(options, interest_rate)

    # Predict the option chain
    predicted_chain = pricing_model.predict_option_chain(
        stock_price=stock_data['Adj Close'].iloc[-1],
        options_chain=options,
        interest_rate=interest_rate,
        historical_volatility=historical_volatility.iloc[-1]
    )

    # Generate trading signals
    signal_generator = SignalGenerator(threshold=0.05)
    signals = signal_generator.generate_trading_signals(predicted_chain)

    # Display the signals
    print("\nTrading Signals:")
    print(signal_generator.format_output(signals))

if __name__ == "__main__":
    main()