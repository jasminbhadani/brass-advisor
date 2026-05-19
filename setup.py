# This runs on Streamlit Cloud startup
# Fetches data and trains model automatically
from fetch_data import fetch_prices
from train_model import train
import os

if not os.path.exists("data/prices.csv"):
    print("No data found — fetching...")
    fetch_prices()

if not os.path.exists("models/copper_model.pkl"):
    print("No model found — training...")
    train()