import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(path="housing.csv"):
    if not os.path.exists(path):
        logging.error("Dataset not found at %s", path)
        sys.exit(1)
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    X = df.drop("price", axis=1)
    y = df["price"]
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_model():
    df = load_data()
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf = RandomForestRegressor(n_estimators=200)
    gb = GradientBoostingRegressor(n_estimators=200)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    logging.info("RandomForest RMSE: %s", mean_squared_error(y_test, rf_pred, squared=False))
    logging.info("RandomForest R2: %s", r2_score(y_test, rf_pred))
    logging.info("GradientBoost RMSE: %s", mean_squared_error(y_test, gb_pred, squared=False))
    logging.info("GradientBoost R2: %s", r2_score(y_test, gb_pred))
    plot_results(y_test, rf_pred, "rf_results.png")
    plot_results(y_test, gb_pred, "gb_results.png")
    joblib.dump(rf, "house_price_rf.pkl")
    joblib.dump(gb, "house_price_gb.pkl")
    joblib.dump(scaler, "scaler.pkl")
    logging.info("Models and scaler saved.")

def plot_results(y_true, y_pred, filename):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Prediction Results")
    plt.savefig(filename)

app = Flask(__name__)
rf_model = None
gb_model = None
scaler = None

def load_models():
    global rf_model, gb_model, scaler
    if os.path.exists("house_price_rf.pkl") and os.path.exists("house_price_gb.pkl"):
        rf_model = joblib.load("house_price_rf.pkl")
        gb_model = joblib.load("house_price_gb.pkl")
        scaler = joblib.load("scaler.pkl")
    else:
        logging.error("Models not found. Train first.")

@app.route("/predict", methods=["POST"])
def predict():
    if rf_model is None or scaler is None:
        return jsonify({"error":"Models not loaded"})
    data = request.json
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    all_features = rf_model.n_features_in_
    X = np.zeros((1, all_features))
    try:
        X_scaled = scaler.transform(df)
    except:
        return jsonify({"error":"Feature mismatch"})
    rf_pred = rf_model.predict(X_scaled)[0]
    gb_pred = gb_model.predict(X_scaled)[0]
    return jsonify({"RandomForest":rf_pred,"GradientBoost":gb_pred})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    else:
        load_models()
        app.run(debug=True)
