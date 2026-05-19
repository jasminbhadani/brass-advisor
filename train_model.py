import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os

def train():
    # Load data
    df = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    
    # Create lag features (past prices as inputs)
    df["copper_lag1"]  = df["copper"].shift(1)
    df["copper_lag7"]  = df["copper"].shift(7)
    df["copper_lag30"] = df["copper"].shift(30)
    df["usdinr_lag1"]  = df["usdinr"].shift(1)
    df["zinc_lag1"]    = df["zinc"].shift(1)
    
    # Target = copper price 7 days from now
    df["target"] = df["copper"].shift(-7)
    df.dropna(inplace=True)

    features = [
        "copper_lag1", "copper_lag7", "copper_lag30",
        "usdinr_lag1", "zinc_lag1"
    ]

    X = df[features]
    y = df["target"]

    # Split — last 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Check accuracy
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    accuracy = round((1 - mape) * 100, 1)
    print(f"✅ Model Accuracy: {accuracy}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/copper_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model saved to models/copper_model.pkl")

    return model, accuracy

if __name__ == "__main__":
    train()