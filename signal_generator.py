import pandas as pd
import numpy as np

class SignalGenerator:
    def __init__(self, price_threshold: float = 0.05, vol_threshold: float = 0.10):
        """
        Initialize the SignalGenerator with thresholds for detecting mispriced options.
        
        :param price_threshold: Percentage difference between predicted and market prices to trigger a signal (default is 5%).
        :param vol_threshold: Percentage difference between predicted and market implied volatility to trigger a signal (default is 10%).
        """
        self.price_threshold = price_threshold
        self.vol_threshold = vol_threshold

    def generate_trading_signals(self, predicted_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals by comparing predicted prices to market prices.
        
        :param predicted_chain: DataFrame containing predicted option prices and market prices.
        :return: DataFrame with trading signals.
        """
        signals = []
        for _, option in predicted_chain.iterrows():
            # Ensure all values are scalar
            predicted_price = float(option['predictedPrice'])
            market_price = float(option['mark'])
            
            # Calculate price difference
            if market_price > 0:
                price_difference = (predicted_price - market_price) / market_price
            else:
                price_difference = 0.0
            
            # Calculate volatility difference
            predicted_vol = float(option['predictedVolatility'])
            market_vol = float(option['marketVolatility']) if not np.isnan(option['marketVolatility']) else predicted_vol
            
            if market_vol > 0:
                vol_difference = (predicted_vol - market_vol) / market_vol
            else:
                vol_difference = 0.0
            
            # Generate signal based on price difference
            if abs(price_difference) > self.price_threshold:
                if price_difference > 0:
                    signal = "Buy"  # Option is undervalued in the market
                else:
                    signal = "Sell"  # Option is overvalued in the market
            else:
                signal = "Hold"  # Option is fairly priced
            
            # Generate signal explanation
            if signal == "Buy":
                explanation = f"Model predicts price is {price_difference*100:.2f}% higher than market. "
                if vol_difference > self.vol_threshold:
                    explanation += f"Market is underestimating volatility by {vol_difference*100:.2f}%."
                elif vol_difference < -self.vol_threshold:
                    explanation += f"Market price doesn't reflect lower volatility expected by model."
                else:
                    explanation += "Other factors contribute to the predicted higher value."
            elif signal == "Sell":
                explanation = f"Model predicts price is {abs(price_difference)*100:.2f}% lower than market. "
                if vol_difference < -self.vol_threshold:
                    explanation += f"Market is overestimating volatility by {abs(vol_difference)*100:.2f}%."
                elif vol_difference > self.vol_threshold:
                    explanation += f"Market price is too high despite higher predicted volatility."
                else:
                    explanation += "Other factors contribute to the predicted lower value."
            else:
                explanation = "Model predicts market price is fair within the specified threshold."
            
            # Format for signal strength
            strength = "Strong" if abs(price_difference) > 2 * self.price_threshold else "Moderate"
            
            # Append signal data
            signals.append({
                "strikePrice": float(option['strikePrice']),
                "expirationDate": str(option['expirationDate']),
                "putCall": str(option['putCall']),
                "marketPrice": market_price,
                "predictedPrice": predicted_price,
                "marketVolatility": market_vol,
                "predictedVolatility": predicted_vol,
                "priceDifference": price_difference,
                "volDifference": vol_difference,
                "signal": signal,
                "signalStrength": strength,
                "explanation": explanation,
                "delta": float(option['delta']),
                "gamma": float(option['gamma']),
                "theta": float(option['theta']),
                "vega": float(option['vega']),
                "rho": float(option['rho']),
                "daysToExpiration": int(option['daysToExpiration']),
                "inTheMoney": bool(option['inTheMoney'])
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
            price_diff_str = f"+{signal['priceDifference']*100:.2f}%" if signal['priceDifference'] >= 0 else f"{signal['priceDifference']*100:.2f}%"
            vol_diff_str = f"+{signal['volDifference']*100:.2f}%" if signal['volDifference'] >= 0 else f"{signal['volDifference']*100:.2f}%"
            
            output.append(
                f"Option: {signal['putCall']} {signal['strikePrice']} "
                f"Expiring in {signal['daysToExpiration']} days ({signal['expirationDate']})\n"
                f"Market Price: ${signal['marketPrice']:.2f}, "
                f"Predicted Price: ${signal['predictedPrice']:.2f}, "
                f"Difference: {price_diff_str}\n"
                f"Market IV: {signal['marketVolatility']*100:.2f}%, "
                f"Predicted IV: {signal['predictedVolatility']*100:.2f}%, "
                f"Difference: {vol_diff_str}\n"
                f"Signal: {signal['signalStrength']} {signal['signal']}\n"
                f"Explanation: {signal['explanation']}\n"
                f"Greeks - Delta: {signal['delta']:.4f}, Gamma: {signal['gamma']:.4f}, "
                f"Theta: {signal['theta']:.4f}, Vega: {signal['vega']:.4f}, Rho: {signal['rho']:.4f}\n"
                f"In The Money: {signal['inTheMoney']}\n"
                f"{'-' * 50}"
            )
        return "\n".join(output)

    def get_top_opportunities(self, signals: pd.DataFrame, top_n: int = 5):
        """
        Get the top trading opportunities based on price difference.
        
        :param signals: DataFrame containing trading signals.
        :param top_n: Number of top opportunities to return.
        :return: DataFrame with top opportunities.
        """
        # Get buy opportunities
        buy_signals = signals[signals['signal'] == 'Buy'].copy()
        if not buy_signals.empty:
            buy_signals = buy_signals.sort_values('priceDifference', ascending=False).head(top_n)
        
        # Get sell opportunities
        sell_signals = signals[signals['signal'] == 'Sell'].copy()
        if not sell_signals.empty:
            sell_signals = sell_signals.sort_values('priceDifference', ascending=True).head(top_n)
        
        # Combine the results
        top_opportunities = pd.concat([buy_signals, sell_signals])
        
        return top_opportunities