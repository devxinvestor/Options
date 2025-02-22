from dotenv import load_dotenv
from time import sleep
import schwabdev
import datetime
import logging
import os
from data_loader import options_chain

def main():
    ticker = input("Enter the ticker symbol: ")
    options_chain = options_chain(ticker)