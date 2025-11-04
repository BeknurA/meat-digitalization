# Meat_Digitalization/ml.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------
# Конфигурация модели
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MODEL_FILE = DATA_DIR / "model.pkl"
SCALER_FILE = DATA_DIR / "scaler.pkl"


class SimplePHModel:
    """
    Класс для обучения и прогнозирования pH с использованием простой линейной регрессии.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        if MODEL_FILE.exists() and SCALER_FILE.exists():
            try:
                self.model = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
            except Exception:
                self.model, self.scaler = None, None

    def train(self, df, target='pH', feature_cols=None):
        """
        Обучает модель на предоставленном DataFrame.
        """
        df = df.copy()
        if target not in df.columns:
            raise ValueError("Целевой столбец не найден")
        df = df.dropna(subset=[target])
        if feature_cols is None or len(feature_cols) == 0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target]

        if len(feature_cols) == 0:
            df['_ones'] = 1.0
            feature_cols = ['_ones']

        X = df[feature_cols].astype(float).values
        y = df[target].astype(float).values

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.model = LinearRegression()
        self.model.fit(Xs, y)

        joblib.dump(self.model, MODEL_FILE)
        joblib.dump(self.scaler, SCALER_FILE)

        preds = self.model.predict(Xs)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) if len(y) > 1 else 0.0
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"rmse": rmse, "r2": r2, "n": int(len(y)), "features": feature_cols}

    def predict(self, df, feature_cols=None):
        """
        Делает прогнозы на основе обученной модели.
        """
        if self.model is None or self.scaler is None:
            n = len(df) if df is not None else 1
            return np.array([6.5] * n)

        if feature_cols is None or len(feature_cols) == 0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(feature_cols) == 0:
            X = np.ones((len(df), 1))
        else:
            X = df[feature_cols].astype(float).values

        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        return np.clip(preds, 0.0, 14.0)


ph_model = SimplePHModel()
