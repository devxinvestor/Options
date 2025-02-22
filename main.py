import schwabdev
import yfinance as yf
from dotenv import load_dotenv
from time import sleep
import datetime
import logging
import os
from data_loader import options_chain
from utils import plot_volatility_surface, plot_historical_volatility
from signal_generator import SignalGenerator
from pricing_model import PricingModel

def main():
    ticker = input("Enter the ticker symbol: ")

    # Fetch options chain and stock data
    options, interest_rate, underlying_volatility = options_chain(ticker)
    stock_data = yf.download(ticker, period="5y")

    # Calculate historical volatility
    pricing_model = PricingModel()
    stock_data['LogReturn'] = pricing_model.calculate_log_returns(stock_data['Adj Close'])
    historical_volatility = pricing_model.calculate_historical_volatility(stock_data['LogReturn']).iloc[-1]

    # Prepare features and labels for training
    options['Moneyness'] = stock_data['Adj Close'].iloc[-1] / options['strikePrice']
    options['TimeToExpiration'] = options['daysToExpiration'] / 365
    options['RiskFreeRate'] = interest_rate / 100
    options['HistoricalVolatility'] = historical_volatility

    # Train the model
    pricing_model.train_xgboost_model(options, interest_rate)

    # Predict the option chain
    predicted_chain = pricing_model.predict_option_chain(
        stock_price=stock_data['Adj Close'].iloc[-1],
        options_chain=options,
        interest_rate=interest_rate,
        historical_volatility=historical_volatility
    )

    # Generate trading signals
    signal_generator = SignalGenerator(threshold=0.02)  # 2% threshold
    signals = signal_generator.generate_trading_signals(predicted_chain)

    # Display the signals
    print("\nTrading Signals:")
    print(signal_generator.format_output(signals))

    plot_volatility_surface(predicted_chain)
    plot_historical_volatility(stock_data)