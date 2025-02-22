import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from math import log, sqrt, exp
from volatility_model import VolatilityModel

class PricingModel:
    def __init__(self):
        self.model = None
        self.volatility_model = VolatilityModel()  # Initialize VolatilityModel

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

    def train_xgboost_model(self, options_chain: pd.DataFrame, interest_rate: float) -> XGBRegressor:
        """
        Train an XGBoost model using features derived from the options chain.
        """
        # Prepare features and labels
        features = options_chain[['Moneyness', 'TimeToExpiration', 'RiskFreeRate', 'HistoricalVolatility']]
        labels = options_chain['ImpliedVolatility']
        
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

    def predict_option_chain(self, stock_price: float, options_chain: pd.DataFrame, interest_rate: float, historical_volatility: float) -> pd.DataFrame:
        """
        Predict the option chain with AI-predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        # Interpolate the volatility surface
        options_chain = self.volatility_model.interpolate_vol_surface(options_chain)
        
        # Prepare the options chain data
        predicted_options = []
        for _, option in options_chain.iterrows():
            # Extract relevant fields
            S = stock_price  # Current stock price
            K = option['strikePrice']
            T = option['daysToExpiration'] / 365  # Convert days to years
            r = interest_rate / 100  # Convert percentage to decimal
            
            # Predict implied volatility using the trained model
            features = pd.DataFrame({
                'Moneyness': [S / K],
                'TimeToExpiration': [T],
                'RiskFreeRate': [r],
                'HistoricalVolatility': [historical_volatility]
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
                "interestRate": interest_rate,
                "marketVolatility": historical_volatility
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