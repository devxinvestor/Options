import os
import logging
import pandas as pd
from time import sleep
from dotenv import load_dotenv
import schwabdev


def options_chain(ticker, contractType="ALL", strategy="SINGLE", range="OTM", optionType="ALL", entitlement="ALL"):
    """
    Get Options Chain
    """
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    client = schwabdev.Client(os.getenv("app_key"), os.getenv("app_secret"), os.getenv("callback_url"))
    options_chain = client.options_chain(symbol=ticker, contractType=contractType, includeUnderlyingQuote=True, strategy=strategy, range=range, optionType=optionType, entitlement=entitlement).json()
    sleep(3)

    interest_rate = options_chain["interestRate"]
    underlying_volatility = options_chain["volatility"]

    options_list = []
    expiration_dates = set()

    for exp_date, strikes in options_chain["callExpDateMap"].items():
        if exp_date.split(":")[0] not in expiration_dates:
            expiration_dates.add(exp_date.split(":")[0])
        for strike, options in strikes.items():
            for option in options:
                options_list.append({
                    "strikePrice": float(strike),
                    "expirationDate": exp_date.split(":")[0],
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
            
    options_list_df = pd.DataFrame(options_list)

    return options_list_df, interest_rate, underlying_volatility