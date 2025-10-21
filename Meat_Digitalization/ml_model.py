# ml_model.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

MODEL_PATH = Path("data") / "model.pkl"
SCALER_PATH = Path("data") / "scaler.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

class MeatPHModel:
    """Simple wrapper: scaler + linear regression. Trainable on CSVs with 'pH' column."""
    def __init__(self):
        self.model = None
        self.scaler = None
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
            except Exception:
                self.model, self.scaler = None, None

    def train(self, df: pd.DataFrame, target_col="pH", feature_cols=None, test_size=0.2, random_state=42):
        """
        Train model on df. If feature_cols is None, use numeric columns excluding target.
        Returns metrics dict and fitted model.
        """
        df = df.copy()
        df = df.dropna(subset=[target_col])
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[float, int]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target_col]
        if len(feature_cols) == 0:
            # fallback: use index/time as single feature so model can run (toy)
            df["_ones"] = 1.0
            feature_cols = ["_ones"]

        X = df[feature_cols].astype(float)
        y = df[target_col].astype(float)

        # scaler
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=test_size, random_state=random_state)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        # save artifacts
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

        return {"rmse": float(rmse), "r2": float(r2), "n_train": int(len(X_train)), "n_test": int(len(X_test)), "features": feature_cols}

    def predict(self, df: pd.DataFrame, feature_cols=None):
        """Predict pH for df. Accepts dataframe of numeric features or will use default behavior."""
        if self.model is None or self.scaler is None:
            # fallback: constant prediction
            n = len(df) if df is not None else 1
            return np.array([6.5] * n)
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        if len(feature_cols) == 0:
            X = np.ones((len(df), 1))
        else:
            X = df[feature_cols].astype(float).values
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        # Ensure pH predictions are in realistic bounds [0,14]
        preds = np.clip(preds, 0.0, 14.0)
        return preds
