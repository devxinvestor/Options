from dotenv import load_dotenv
from time import sleep
import schwabdev
import datetime
import logging
import os

def options_chain(ticker, contractType="ALL", strategy="SINGLE", range="OTM", optionType="ALL", entitlement="ALL"):
    """
    Get Options Chain
    """
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    client = schwabdev.Client(os.getenv("app_key"), os.getenv("app_secret"), os.getenv("callback_url"))

    options_chain = options_chain(symbol=ticker, contractType=contractType, includeUnderlyingQuote=True, strategy=strategy, range=range, optionType=optionType, entitlement=entitlement).json()
    sleep(3)

    expiration_chain = client.option_expiration_chain(ticker).json()
    sleep(3)

