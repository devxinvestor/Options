import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from math import log, sqrt, exp

class PricingModel:
    def __init__(self):
        self.model = None
        self.feature_importance = None

    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate the Black-Scholes price for an option.
        """
        if T <= 0:
            # Handle case when time to expiration is zero or negative
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
                
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if option_type == 'call':
            return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:
            return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def prepare_features(self, options_chain: pd.DataFrame, stock_price: float, interest_rate: float, historical_volatility: float) -> pd.DataFrame:
        """
        Prepare features for the XGBoost model.
        """
        # Calculate moneyness (S/K)
        options_chain['Moneyness'] = stock_price / options_chain['strikePrice']
        
        # Convert days to expiration to years
        options_chain['TimeToExpiration'] = options_chain['daysToExpiration'] / 365.0
        
        # Add risk-free rate
        options_chain['RiskFreeRate'] = interest_rate / 100.0
        
        # Add historical volatility
        options_chain['HistoricalVolatility'] = historical_volatility
        
        # Add option type (call=1, put=0)
        options_chain['OptionType'] = options_chain['putCall'].apply(lambda x: 1 if x.lower() == 'call' else 0)
        
        # Add bid-ask spread
        options_chain['BidAskSpread'] = options_chain['ask'] - options_chain['bid']
        options_chain['BidAskRatio'] = (options_chain['ask'] - options_chain['bid']) / ((options_chain['ask'] + options_chain['bid']) / 2)
        
        # Add distance from ATM
        options_chain['DistanceFromATM'] = abs(1 - options_chain['Moneyness'])
        
        # Calculate theoretical Black-Scholes price using historical volatility
        options_chain['BSPrice'] = options_chain.apply(
            lambda row: self.black_scholes_price(
                stock_price, 
                row['strikePrice'],
                row['TimeToExpiration'],
                row['RiskFreeRate'],
                historical_volatility,
                row['putCall'].lower()
            ), axis=1)
        
        # Calculate price difference from BS model
        options_chain['MarketVsBS'] = (options_chain['mark'] - options_chain['BSPrice']) / options_chain['BSPrice']
        
        # Open interest and volume-based features
        options_chain['OpenInterestNorm'] = np.log1p(options_chain['openInterest'])
        
        return options_chain

    def train_xgboost_model(self, options_chain: pd.DataFrame, stock_price: float, interest_rate: float, historical_volatility: float) -> XGBRegressor:
        """
        Train an XGBoost model using features derived from the options chain.
        """
        # Prepare features
        options_data = self.prepare_features(options_chain.copy(), stock_price, interest_rate, historical_volatility)
        
        # Remove rows with missing implied volatility
        options_data = options_data.dropna(subset=['ImpliedVolatility'])
        
        # Select features for training
        features = options_data[[
            'Moneyness', 'TimeToExpiration', 'RiskFreeRate', 'HistoricalVolatility',
            'OptionType', 'BidAskSpread', 'BidAskRatio', 'DistanceFromATM', 
            'BSPrice', 'MarketVsBS', 'OpenInterestNorm', 'strikePrice'
        ]]
        
        # Target variable is the implied volatility
        labels = options_data['ImpliedVolatility']
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # Initialize and train the XGBoost model
        self.model = XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Validate the model
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Validation Mean Squared Error: {mse:.6f}")
        print(f"Validation Root Mean Squared Error: {rmse:.6f}")
        print(f"Validation R² Score: {r2:.6f}")
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return self.model

    def predict_implied_volatility(self, options_data: pd.DataFrame, stock_price: float, interest_rate: float, historical_volatility: float) -> pd.DataFrame:
        """
        Predict implied volatility using the trained XGBoost model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        # Prepare features for prediction
        options_features = self.prepare_features(options_data.copy(), stock_price, interest_rate, historical_volatility)
        
        # Select features for prediction
        X_pred = options_features[[
            'Moneyness', 'TimeToExpiration', 'RiskFreeRate', 'HistoricalVolatility',
            'OptionType', 'BidAskSpread', 'BidAskRatio', 'DistanceFromATM', 
            'BSPrice', 'MarketVsBS', 'OpenInterestNorm', 'strikePrice'
        ]]
        
        # Predict implied volatility
        options_features['PredictedIV'] = self.model.predict(X_pred)
        
        return options_features

    def predict_option_chain(self, stock_price: float, options_chain: pd.DataFrame, interest_rate: float, historical_volatility: float) -> pd.DataFrame:
        """
        Predict the option chain with AI-predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        # Predict implied volatility using the trained model
        options_with_predicted_iv = self.predict_implied_volatility(
            options_chain.copy(), stock_price, interest_rate, historical_volatility
        )
        
        # Prepare the options chain data with predicted prices
        predicted_options = []
        for _, option in options_with_predicted_iv.iterrows():
            # Extract relevant fields
            S = stock_price  # Current stock price
            K = option['strikePrice']
            T = option['TimeToExpiration']  # Already converted to years
            r = option['RiskFreeRate']  # Already converted to decimal
            predicted_iv = option['PredictedIV']  # Use AI-predicted implied volatility
            option_type = option['putCall'].lower()
            
            # Calculate theoretical option value using the predicted IV
            theoretical_price = self.black_scholes_price(S, K, T, r, predicted_iv, option_type)
            
            option_data = {
                "strikePrice": K,
                "expirationDate": option['expirationDate'],
                "putCall": option['putCall'],
                "bid": option['bid'],
                "ask": option['ask'],
                "mark": option['mark'],  # Keep original market price
                "marketVolatility": option['ImpliedVolatility'],  # Original implied volatility
                "predictedVolatility": predicted_iv,  # AI-predicted implied volatility
                "predictedPrice": theoretical_price,  # Theoretical price based on predicted IV
                "delta": self.calculate_greeks(S, K, T, r, predicted_iv, 'delta', option_type),
                "gamma": self.calculate_greeks(S, K, T, r, predicted_iv, 'gamma', option_type),
                "theta": self.calculate_greeks(S, K, T, r, predicted_iv, 'theta', option_type),
                "vega": self.calculate_greeks(S, K, T, r, predicted_iv, 'vega', option_type),
                "rho": self.calculate_greeks(S, K, T, r, predicted_iv, 'rho', option_type),
                "openInterest": option['openInterest'],
                "timeValue": option['timeValue'] if 'timeValue' in option else 0.0,
                "daysToExpiration": option['daysToExpiration'],
                "inTheMoney": option['inTheMoney'],
                "priceDifference": (theoretical_price - option['mark']) / option['mark'] if option['mark'] > 0 else 0
            }
            predicted_options.append(option_data)
        
        return pd.DataFrame(predicted_options)

    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, greek: str, option_type: str) -> float:
        """
        Calculate Black-Scholes Greeks.
        """
        if T <= 0:
            # Handle case when time to expiration is zero
            if greek == 'delta':
                if option_type == 'call':
                    return 1.0 if S > K else 0.0
                else:
                    return -1.0 if S < K else 0.0
            # For other Greeks, return 0 when T is 0
            return 0.0
            
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if greek == 'delta':
            return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        elif greek == 'gamma':
            return norm.pdf(d1) / (S * sigma * sqrt(T))
        elif greek == 'theta':
            term1 = -S * norm.pdf(d1) * sigma / (2 * sqrt(T))
            term2 = r * K * exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -r * K * exp(-r * T) * norm.cdf(-d2)
            return (term1 - term2) / 365  # Converting to daily theta
        elif greek == 'vega':
            return S * norm.pdf(d1) * sqrt(T) / 100  # Scaled by 100 for 1% change in volatility
        elif greek == 'rho':
            return K * T * exp(-r * T) * norm.cdf(d2) / 100 if option_type == 'call' else -K * T * exp(-r * T) * norm.cdf(-d2) / 100
        else:
            raise ValueError(f"Unsupported Greek: {greek}")