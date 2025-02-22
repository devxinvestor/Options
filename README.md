# ML Options Pricing Model Using XGBoost and GARCH

## Overview
This project implements a machine learning-based options pricing model using **XGBoost** for predicting option prices and **GARCH** for volatility estimation. The model leverages historical stock price data and options chain data to predict option prices, calculate implied volatility, and generate trading signals.

The project is designed to:
1. Retrieve historical stock prices and options chain data.
2. Estimate volatility using GARCH models.
3. Train an XGBoost model to predict option prices.
4. Generate trading signals based on predicted vs. market prices.
5. Provide a user-friendly interface for analyzing options.

---

## Features
- **Data Retrieval**: Fetch historical stock prices and options chain data using APIs like `yfinance` and `schwabdev`.
- **Volatility Estimation**: Use GARCH models to estimate historical and implied volatility.
- **Options Pricing**: Train an XGBoost model to predict option prices and calculate Greeks (Delta, Gamma, Theta, Vega, Rho).
- **Trading Signals**: Compare predicted prices to market prices and generate Buy/Sell/Hold signals.
- **Visualization**: Plot volatility surfaces and price charts for better insights.

---

## Project Structure
ml-options-pricing/
├── data_loader.py           # Handles data retrieval (stock prices, options chain)
├── volatility_model.py      # Implements GARCH and volatility surface interpolation  
├── pricing_model.py         # Trains XGBoost model and predicts option prices
├── signal_generator.py      # Generates trading signals based on predicted prices
├── utils.py                 # Helper functions (file I/O, visualization, etc.)
├── main.py                  # Entry point for the application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (API keys, etc.)

---

## Example Output

```plaintext
Trading Signals:
--------------------------------------------------
Option: CALL 150.0 Expiring on 2023-12-15
Market Price: $5.25, Predicted Price: $5.75, Difference: 9.52%
Signal: Buy
Greeks - Delta: 0.6234, Gamma: 0.0234, Theta: -0.0456, Vega: 0.1234, Rho: 0.0345
In The Money: True
--------------------------------------------------
Option: PUT 160.0 Expiring on 2023-12-15
Market Price: $4.80, Predicted Price: $4.50, Difference: 6.25%
Signal: Sell
Greeks - Delta: -0.4234, Gamma: 0.0187, Theta: -0.0321, Vega: 0.0987, Rho: -0.0234
In The Money: False
--------------------------------------------------
```

## Modules

### 1. `data_loader.py`
#### Purpose: Fetches historical stock prices and options chain data.

#### Key Functions:
- `get_stock_data(ticker: str) -> pd.DataFrame`: Retrieves historical stock prices using `yfinance`.
- `get_options_chain(ticker: str) -> dict`: Retrieves the current options chain using `schwabdev`.

---

### 2. `volatility_model.py`
#### Purpose: Estimates volatility using GARCH models and interpolates the volatility surface.

#### Key Functions:
- `calculate_garch_volatility(price_data: pd.Series) -> pd.Series`: Fits a GARCH model to estimate volatility.
- `interpolate_vol_surface(options_data: pd.DataFrame) -> pd.DataFrame`: Interpolates the volatility surface.

---

### 3. `pricing_model.py`
#### Purpose: Trains an XGBoost model to predict option prices and calculates Greeks.

#### Key Functions:
- `train_xgboost_model(features: pd.DataFrame, labels: pd.Series) -> XGBRegressor`: Trains the XGBoost model.
- `predict_option_chain(stock_price: float, options_chain: pd.DataFrame, interest_rate: float, historical_volatility: float) -> pd.DataFrame`: Predicts option prices and calculates Greeks.

---

### 4. `signal_generator.py`
#### Purpose: Generates trading signals by comparing predicted prices to market prices.

#### Key Functions:
- `generate_trading_signals(predicted_chain: pd.DataFrame) -> pd.DataFrame`: Generates Buy/Sell/Hold signals.
- `format_output(signals: pd.DataFrame) -> str`: Formats the trading signals for display.

---

### 5. `utils.py`
#### Purpose: Provides helper functions for file I/O, visualization, and data preprocessing.

#### Key Functions:
- `save_results_to_csv(results: pd.DataFrame, filename: str) -> None`: Saves results to a CSV file.
- `plot_volatility_surface(vol_data: pd.DataFrame) -> None`: Plots the volatility surface.

---