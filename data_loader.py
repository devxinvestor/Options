import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from math import log, sqrt, exp
from typing import Dict, List

class PricingModel:
    def __init__(self):
        self.model = None

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate log returns from a series of prices.
        """
        return np.log(prices / prices.shift(1))

    def calculate_historical_volatility(self, log_returns: pd.Series, window: int = 21) -> pd.Series:
        """
        Calculate rolling historical volatility from log returns.
        """
        return log_returns.rolling(window=window).std() * np.sqrt(252)

    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate the Black-Scholes price for an option.
        """
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:
            return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def black_scholes_implied_volatility(self, S: float, K: float, T: float, r: float, price: float, option_type: str = 'call') -> float:
        """
        Calculate implied volatility using the Black-Scholes model.
        """
        def bs_price(sigma):
            d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            if option_type == 'call':
                return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            else:
                return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Use Newton-Raphson method to find implied volatility
        sigma = 0.5  # Initial guess
        for _ in range(100):
            price_diff = bs_price(sigma) - price
            if abs(price_diff) < 1e-6:
                break
            sigma = sigma - price_diff / (S * sqrt(T) * norm.pdf((log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))))
        return sigma

    def train_xgboost_model(self, features: pd.DataFrame, labels: pd.Series) -> XGBRegressor:
        """
        Train an XGBoost model on historical data.
        """
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # Initialize and train the XGBoost model
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
        self.model.fit(X_train, y_train)
        
        # Validate the model
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        print(f"Validation Mean Squared Error: {mse}")
        
        return self.model

    def predict_option_chain(self, stock_data: pd.DataFrame, options_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the option chain with AI-predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        # Prepare the options chain data
        predicted_options = []
        for _, option in options_chain.iterrows():
            # Extract relevant fields
            S = stock_data['Adj Close'].iloc[-1]  # Current stock price
            K = option['strikePrice']
            T = option['daysToExpiration'] / 365  # Convert days to years
            r = option['interestRate'] / 100  # Convert percentage to decimal
            market_volatility = option['marketVolatility']
            
            # Predict implied volatility using the trained model
            features = pd.DataFrame({
                'RollingVolatility': [market_volatility],
                'Moneyness': [S / K],
                'TimeToExpiration': [T],
                'RiskFreeRate': [r]
            })
            predicted_iv = self.model.predict(features)[0]
            
            # Calculate theoretical option value and Greeks
            theoretical_price = self.black_scholes_price(S, K, T, r, predicted_iv, option['putCall'].lower())
            option_data = {
                "strikePrice": K,
                "expirationDate": option['expirationDate'],
                "putCall": option['putCall'],
                "bid": option['bid'],
                "ask": option['ask'],
                "mark": theoretical_price,  # Use predicted price as mark
                "volatility": predicted_iv,  # Use predicted implied volatility
                "delta": self.calculate_greeks(S, K, T, r, predicted_iv, 'delta', option['putCall'].lower()),
                "gamma": self.calculate_greeks(S, K, T, r, predicted_iv, 'gamma', option['putCall'].lower()),
                "theta": self.calculate_greeks(S, K, T, r, predicted_iv, 'theta', option['putCall'].lower()),
                "vega": self.calculate_greeks(S, K, T, r, predicted_iv, 'vega', option['putCall'].lower()),
                "rho": self.calculate_greeks(S, K, T, r, predicted_iv, 'rho', option['putCall'].lower()),
                "openInterest": option['openInterest'],
                "timeValue": option['timeValue'],
                "theoreticalOptionValue": theoretical_price,
                "theoreticalVolatility": predicted_iv,
                "daysToExpiration": option['daysToExpiration'],
                "inTheMoney": option['inTheMoney'],
                "interestRate": option['interestRate'],
                "marketVolatility": option['marketVolatility']
            }
            predicted_options.append(option_data)
        
        return pd.DataFrame(predicted_options)

    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, greek: str, option_type: str) -> float:
        """
        Calculate Black-Scholes Greeks.
        """
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if greek == 'delta':
            return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        elif greek == 'gamma':
            return norm.pdf(d1) / (S * sigma * sqrt(T))
        elif greek == 'theta':
            term1 = -S * norm.pdf(d1) * sigma / (2 * sqrt(T))
            term2 = r * K * exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -r * K * exp(-r * T) * norm.cdf(-d2)
            return (term1 + term2) / 365
        elif greek == 'vega':
            return S * norm.pdf(d1) * sqrt(T) / 100
        elif greek == 'rho':
            return K * T * exp(-r * T) * norm.cdf(d2) / 100 if option_type == 'call' else -K * T * exp(-r * T) * norm.cdf(-d2) / 100
        else:
            raise ValueError(f"Unsupported Greek: {greek}")