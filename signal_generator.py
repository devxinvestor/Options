import pandas as pd

class SignalGenerator:
    def __init__(self, threshold: float = 0.05):
        """
        Initialize the SignalGenerator with a threshold for detecting mispriced options.
        
        :param threshold: Percentage difference between predicted and market prices to trigger a signal (default is 5%).
        """
        self.threshold = threshold

    def calculate_greeks(self, option_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Black-Scholes Greeks for the options.
        
        :param option_data: DataFrame containing option data.
        :return: DataFrame with Greeks added.
        """
        # Greeks are already calculated in the PricingModel, so no need to recalculate here.
        return option_data

    def generate_trading_signals(self, predicted_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals by comparing predicted prices to market prices.
        
        :param predicted_chain: DataFrame containing predicted option prices and market prices.
        :return: DataFrame with trading signals.
        """
        signals = []
        for _, option in predicted_chain.iterrows():
            predicted_price = option['theoreticalOptionValue']
            market_price = option['mark']
            price_difference = abs(predicted_price - market_price) / market_price
            
            if price_difference > self.threshold:
                if predicted_price > market_price:
                    signal = "Buy"  # Option is undervalued
                else:
                    signal = "Sell"  # Option is overvalued
            else:
                signal = "Hold"  # Option is fairly priced
            
            signals.append({
                "strikePrice": option['strikePrice'],
                "expirationDate": option['expirationDate'],
                "putCall": option['putCall'],
                "marketPrice": market_price,
                "predictedPrice": predicted_price,
                "priceDifference": price_difference,
                "signal": signal,
                "delta": option['delta'],
                "gamma": option['gamma'],
                "theta": option['theta'],
                "vega": option['vega'],
                "rho": option['rho'],
                "inTheMoney": option['inTheMoney']
            })
        
        return pd.DataFrame(signals)

    def format_output(self, signals: pd.DataFrame) -> str:
        """
        Format the trading signals into a user-friendly output.
        
        :param signals: DataFrame containing trading signals.
        :return: Formatted string for display.
        """
        output = []
        for _, signal in signals.iterrows():
            output.append(
                f"Option: {signal['putCall']} {signal['strikePrice']} "
                f"Expiring on {signal['expirationDate']}\n"
                f"Market Price: ${signal['marketPrice']:.2f}, "
                f"Predicted Price: ${signal['predictedPrice']:.2f}, "
                f"Difference: {signal['priceDifference'] * 100:.2f}%\n"
                f"Signal: {signal['signal']}\n"
                f"Greeks - Delta: {signal['delta']:.4f}, Gamma: {signal['gamma']:.4f}, "
                f"Theta: {signal['theta']:.4f}, Vega: {signal['vega']:.4f}, Rho: {signal['rho']:.4f}\n"
                f"In The Money: {signal['inTheMoney']}\n"
                f"{'-' * 50}"
            )
        return "\n".join(output)