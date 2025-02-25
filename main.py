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

    # Calculate log returns for GARCH model
    stock_data['LogReturn'] = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
    stock_data.dropna(inplace=True)

    # Initialize VolatilityModel
    underlying_price = stock_data['Adj Close'].iloc[-1]
    volatility_model = VolatilityModel(options, underlying_price, interest_rate / 100)

    # Fit GARCH model to historical returns
    volatility_model.fit_garch(stock_data['LogReturn'])

    # Calculate implied volatilities for the options chain
    volatility_model.calculate_implied_volatilities()

    # Interpolate the volatility surface
    volatility_model.interpolate_volatility_surface()

    # Add implied volatility and other features to the options chain
    options = volatility_model.get_implied_volatilities()
    options['Moneyness'] = underlying_price / options['strikePrice']
    options['TimeToExpiration'] = options['daysToExpiration'] / 365
    options['RiskFreeRate'] = interest_rate / 100
    options['HistoricalVolatility'] = volatility_model.garch_model_fit.conditional_volatility[-1]

    # Train the XGBoost model
    pricing_model = PricingModel()
    pricing_model.train_xgboost_model(options, interest_rate)

    # Predict the option chain
    predicted_chain = pricing_model.predict_option_chain(
        stock_price=underlying_price,
        options_chain=options,
        interest_rate=interest_rate,
        historical_volatility=volatility_model.garch_model_fit.conditional_volatility[-1]
    )

    # Generate trading signals
    signal_generator = SignalGenerator(threshold=0.05)
    signals = signal_generator.generate_trading_signals(predicted_chain)

    # Display the signals
    print("\nTrading Signals:")
    print(signal_generator.format_output(signals))

if __name__ == "__main__":
    main()