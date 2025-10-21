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

# === Пути к файлам модели ===
MODEL_PATH = Path("data") / "model.pkl"
SCALER_PATH = Path("data") / "scaler.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

class MeatPHModel:
    """
    Модель прогнозирования pH.
    Простая обёртка над LinearRegression + StandardScaler.
    Поддерживает загрузку, обучение, сохранение и предсказания.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = []
        self._load_existing()

    # --- загрузка существующей модели, если есть ---
    def _load_existing(self):
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                print("✅ Модель успешно загружена из файла.")
            except Exception as e:
                print("⚠️ Ошибка загрузки модели:", e)
                self.model, self.scaler = None, None
        else:
            print("ℹ️ Модель не найдена — потребуется обучение.")

    # --- обучение модели ---
    def train(self, df: pd.DataFrame, target_col="pH", feature_cols=None, test_size=0.2, random_state=42):
        """
        Обучает линейную модель на наборе данных с колонкой pH.
        Возвращает метрики RMSE, R² и список признаков.
        """
        df = df.copy()
        df = df.dropna(subset=[target_col])
        if df.empty:
            raise ValueError("Нет данных для обучения (пустой DataFrame)")

        # выбор признаков
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[float, int]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target_col]

        if not feature_cols:
            df["_const"] = 1.0
            feature_cols = ["_const"]

        X = df[feature_cols].astype(float)
        y = df[target_col].astype(float)

        # защита от NaN и отрицательных значений
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.clip(lower=0.0, upper=14.0)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=test_size, random_state=random_state
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        self.feature_cols = feature_cols

        # сохранение артефактов
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

        print(f"✅ Модель обучена. RMSE={rmse:.4f}, R²={r2:.4f}")

        return {
            "rmse": float(rmse),
            "r2": float(r2),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": feature_cols
        }

    # --- предсказание ---
    def predict(self, df: pd.DataFrame, feature_cols=None):
        """
        Делает предсказания pH для данных.
        Если модель не обучена — возвращает 6.5 по умолчанию.
        """
        if self.model is None or self.scaler is None:
            print("⚠️ Модель не обучена — возвращаются константные значения (6.5).")
            n = len(df) if df is not None else 1
            return np.array([6.5] * n)

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        if not feature_cols:
            X = np.ones((len(df), 1))
        else:
            X = df[feature_cols].astype(float).values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        preds = np.clip(preds, 0.0, 14.0)
        return preds

    # --- вспомогательная функция ---
    def evaluate_on_df(self, df, target_col="pH"):
        """Проверяет качество предсказаний на переданном DataFrame."""
        if self.model is None or self.scaler is None:
            return {"error": "Модель не обучена"}
        if target_col not in df.columns:
            return {"error": f"В DataFrame нет колонки {target_col}"}

        feature_cols = [c for c in df.select_dtypes(include=[float, int]).columns if c != target_col]
        if not feature_cols:
            return {"error": "Нет числовых признаков для оценки"}

        X = df[feature_cols].astype(float)
        y_true = df[target_col].astype(float)
        y_pred = self.predict(df, feature_cols)

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return {"rmse": rmse, "r2": r2, "n": len(df)}

    def reset_model(self):
        """Удаляет сохранённую модель и сбрасывает состояние."""
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        if SCALER_PATH.exists():
            SCALER_PATH.unlink()
        self.model, self.scaler = None, None
        print("🔁 Модель инициализирована заново (артефакты удалены).")
