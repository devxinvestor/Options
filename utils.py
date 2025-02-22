import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_volatility_surface(vol_data: pd.DataFrame) -> None:
    """
    Plot the volatility surface.
    
    :param vol_data: DataFrame containing volatility data.
    """
    pivot_table = vol_data.pivot(index='strikePrice', columns='expirationDate', values='volatility')
    sns.heatmap(pivot_table, annot=True, fmt=".2f")
    plt.title("Volatility Surface")
    plt.xlabel("Expiration Date")
    plt.ylabel("Strike Price")
    plt.show()

def plot_price_chart(prices: pd.Series, title: str = "Price Chart") -> None:
    """
    Plot a price chart.
    
    :param prices: Series of prices.
    :param title: Title of the chart.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(prices, label="Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()