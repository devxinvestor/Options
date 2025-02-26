import os
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from data_loader import get_options_chain, parse_options_chain
from pricing_model import PricingModel
from signal_generator import SignalGenerator
from volatility_model import VolatilityModel
from utils import plot_volatility_surface, plot_price_chart

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("options_ai.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("options_ai")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Options AI - ML-based Options Pricing and Signal Generator')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--interest-rate', type=float, default=4.5, help='Risk-free interest rate (annualized)')
    parser.add_argument('--hist-volatility', type=float, default=0.25, help='Historical volatility')
    parser.add_argument('--price-threshold', type=float, default=0.05, help='Price difference threshold for signals')
    parser.add_argument('--vol-threshold', type=float, default=0.10, help='Volatility difference threshold for signals')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top opportunities to display')
    parser.add_argument('--plot', action='store_true', help='Enable data visualization')
    parser.add_argument('--option-type', type=str, default='ALL', choices=['CALL', 'PUT', 'ALL'], 
                        help='Option type to analyze')
    parser.add_argument('--range', type=str, default='OTM', 
                        choices=['ITM', 'OTM', 'ATM', 'NTM', 'ALL'], 
                        help='Option range to analyze')
    parser.add_argument('--output', type=str, default='console', 
                        choices=['console', 'csv', 'json', 'excel'], 
                        help='Output format')
    
    return parser.parse_args()

def get_stock_price(options_chain):
    """Extract the current stock price from the options chain."""
    return options_chain["underlying"]["mark"]

def main():
    """Main function to run the options pricing and signal generation process."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Options AI")
    
    # Parse command line arguments
    args = parse_arguments()
    logger.info(f"Analyzing options for {args.ticker}")
    
    # Fetch options chain data
    try:
        logger.info("Fetching options chain data...")
        options_chain_raw = get_options_chain(
            args.ticker, 
            contractType=args.option_type, 
            range=args.range
        )
        
        # Get current stock price
        stock_price = get_stock_price(options_chain_raw)
        logger.info(f"Current {args.ticker} price: ${stock_price:.2f}")
        
        # Parse options chain
        options_df = parse_options_chain(options_chain_raw)
        logger.info(f"Loaded {len(options_df)} options contracts")
        
        # Initialize volatility model and calculate implied volatilities
        logger.info("Calculating implied volatilities...")
        vol_model = VolatilityModel(options_df, stock_price, args.interest_rate / 100.0)
        vol_model.calculate_implied_volatilities()
        options_df = vol_model.get_implied_volatilities()
        
        if args.plot:
            try:
                vol_model.interpolate_volatility_surface()
                # Create a pivot table for the volatility surface plot
                vol_surface_df = options_df[['strikePrice', 'expirationDate', 'ImpliedVolatility']].copy()
                vol_surface_df.rename(columns={'ImpliedVolatility': 'volatility'}, inplace=True)
                plot_volatility_surface(vol_surface_df)
            except Exception as e:
                logger.error(f"Error plotting volatility surface: {e}")
        
        # Initialize and train the pricing model
        logger.info("Training the pricing model...")
        pricing_model = PricingModel()
        pricing_model.train_xgboost_model(options_df, stock_price, args.interest_rate, args.hist_volatility)
        
        # Print feature importance
        logger.info("Feature importance:")
        for _, row in pricing_model.feature_importance.iterrows():
            logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
        
        # Predict option prices and implied volatilities
        logger.info("Generating predictions...")
        predicted_chain = pricing_model.predict_option_chain(
            stock_price, options_df, args.interest_rate, args.hist_volatility
        )
        
        # Generate trading signals
        logger.info("Generating trading signals...")
        signal_gen = SignalGenerator(
            price_threshold=args.price_threshold, 
            vol_threshold=args.vol_threshold
        )
        signals = signal_gen.generate_trading_signals(predicted_chain)
        
        # Get and print top opportunities
        top_opps = signal_gen.get_top_opportunities(signals, args.top_n)
        logger.info("\n" + signal_gen.format_output(top_opps))
        
        # Save outputs based on specified format
        output_filename = f"{args.ticker}_options_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.output == 'csv':
            signals.to_csv(f"{output_filename}.csv", index=False)
            logger.info(f"Saved signals to {output_filename}.csv")
        elif args.output == 'json':
            signals.to_json(f"{output_filename}.json", orient='records')
            logger.info(f"Saved signals to {output_filename}.json")
        elif args.output == 'excel':
            with pd.ExcelWriter(f"{output_filename}.xlsx") as writer:
                signals.to_excel(writer, sheet_name='Signals', index=False)
                predicted_chain.to_excel(writer, sheet_name='Predictions', index=False)
                options_df.to_excel(writer, sheet_name='Raw Data', index=False)
            logger.info(f"Saved data to {output_filename}.xlsx")
            
        logger.info("Analysis complete")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())