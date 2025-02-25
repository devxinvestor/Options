import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import griddata
from arch import arch_model

class VolatilityModel:
    def __init__(self, options_chain_df, underlying_price, risk_free_rate):
        """
        Initialize the volatility model.

        :param options_chain_df: DataFrame containing options chain data.
        :param underlying_price: Current price of the underlying asset.
        :param risk_free_rate: Risk-free interest rate (annualized).
        """
        self.options_chain_df = options_chain_df
        self.underlying_price = underlying_price
        self.risk_free_rate = risk_free_rate
        self.garch_model = None
        self.volatility_surface = None

    def fit_garch(self, returns):
        """
        Fit a GARCH(1,1) model to the returns of the underlying asset.

        :param returns: A pandas Series or numpy array of log returns.
        """
        self.garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        self.garch_model_fit = self.garch_model.fit(disp='off')

    def calculate_implied_volatility(self, option_price, strike, expiration_days, option_type='call'):
        """
        Calculate implied volatility using the Black-Scholes formula.

        :param option_price: Market price of the option.
        :param strike: Strike price of the option.
        :param expiration_days: Days to expiration.
        :param option_type: 'call' or 'put'.
        :return: Implied volatility.
        """
        def black_scholes_price(volatility):
            # Black-Scholes formula
            S = self.underlying_price
            K = strike
            T = expiration_days / 365.0
            r = self.risk_free_rate
            d1 = (np.log(S / K) + (r + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
            d2 = d1 - volatility * np.sqrt(T)
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return price

        # Minimize the difference between market price and Black-Scholes price
        def objective(volatility):
            return abs(black_scholes_price(volatility) - option_price)

        # Initial guess for volatility
        initial_vol = 0.2
        result = minimize(objective, initial_vol, bounds=[(0.01, 5.0)])
        return result.x[0]

    def calculate_implied_volatilities(self):
        """
        Calculate implied volatilities for all options in the chain.
        """
        implied_vols = []
        for _, row in self.options_chain_df.iterrows():
            option_price = row['mark']  # Use mid-price (average of bid and ask)
            strike = row['strikePrice']
            expiration_days = row['daysToExpiration']
            option_type = row['putCall'].lower()
            iv = self.calculate_implied_volatility(option_price, strike, expiration_days, option_type)
            implied_vols.append(iv)
        self.options_chain_df['ImpliedVolatility'] = implied_vols

    def interpolate_volatility_surface(self):
        """
        Interpolate the volatility surface using cubic spline interpolation.
        """
        strikes = self.options_chain_df['strikePrice'].values
        expirations = self.options_chain_df['daysToExpiration'].values
        implied_vols = self.options_chain_df['ImpliedVolatility'].values

        # Create a grid for interpolation
        grid_strikes, grid_expirations = np.meshgrid(
            np.linspace(strikes.min(), strikes.max(), 100),
            np.linspace(expirations.min(), expirations.max(), 100)
        )

        # Interpolate using cubic spline
        self.volatility_surface = griddata(
            (strikes, expirations), implied_vols,
            (grid_strikes, grid_expirations), method='cubic'
        )

    def get_volatility_surface(self):
        """
        Return the interpolated volatility surface.
        """
        return self.volatility_surface

    def get_implied_volatilities(self):
        """
        Return the DataFrame with implied volatilities.
        """
        return self.options_chain_df