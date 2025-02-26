import os
import logging
import pandas as pd
from time import sleep
from dotenv import load_dotenv
import schwabdev


def get_options_chain(ticker, contractType="ALL", strategy="SINGLE", range="OTM", optionType="ALL"):
    """
    Fetch the options chain data from Schwab API.
    """
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    client = schwabdev.Client(os.getenv("app_key"), os.getenv("app_secret"), os.getenv("callback_url"))
    
    options_chain = client.option_chains(
        symbol=ticker,
        contractType=contractType,
        includeUnderlyingQuote=True,
        strategy=strategy,
        range=range,
        optionType=optionType
    ).json()
    
    sleep(3)  # Prevent rate-limiting issues
    return options_chain


def parse_options_chain(options_chain):
    """
    Parse the options chain data and return a structured DataFrame.
    """
    options_list = []
    expiration_dates = set()

    for exp_date, strikes in options_chain.get("callExpDateMap", {}).items():
        exp_date_clean = exp_date.split(":")[0]
        expiration_dates.add(exp_date_clean)
        
        for strike, options in strikes.items():
            for option in options:
                options_list.append({
                    "strikePrice": float(strike),
                    "expirationDate": exp_date_clean,
                    "putCall": option["putCall"],
                    "bid": option["bid"],
                    "ask": option["ask"],
                    "mark": option["mark"],
                    "volatility": option["volatility"],
                    "delta": option["delta"],
                    "gamma": option["gamma"],
                    "theta": option["theta"],
                    "vega": option["vega"],
                    "rho": option["rho"],
                    "openInterest": option["openInterest"],
                    "timeValue": option["timeValue"],
                    "theoreticalOptionValue": option["theoreticalOptionValue"],
                    "ImpliedVolatility": option["theoreticalVolatility"],
                    "daysToExpiration": option["daysToExpiration"],
                    "inTheMoney": option["inTheMoney"],
                })

    return pd.DataFrame(options_list)