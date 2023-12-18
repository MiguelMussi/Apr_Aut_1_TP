# ---------------------- LIBRERIAS -------------------------

# Data
import pandas as pd
import numpy as np
from datetime import datetime

# Preprocessing data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning Pipeline & process
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import keras
from keras.models import load_model
# from tensorflow.keras.models import load_model


# ---------------------- CLASES -------------------------

class DateExtraction(BaseEstimator, TransformerMixin):
    """
    Extrae el atributo "MES" de la fecha
    y lo recategoriza de manera circular con seno y coseno
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Asegurar que X es un DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['Date'])

        # Convertir la columna 'Date' a datetime
        X['Date'] = pd.to_datetime(X['Date'])

        # Extraer el mes y calcular las funciones sin y cos
        X['Month_sin'] = np.sin(2 * np.pi * X['Date'].dt.month / 12).round(5)
        X['Month_cos'] = np.cos(2 * np.pi * X['Date'].dt.month / 12).round(5)

        # Eliminar la columna original 'Date'
        X = X.drop(['Date'], axis=1)

        # Devolver un array de NumPy X.values
        return X


class CoordRecat(BaseEstimator, TransformerMixin):
    """
    Recategoriza los valores de la lista de coordenadas. 
    Primero a grados sexagesimales, a radianes y luego
    de manera circular con seno y coseno
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mapeo_coord = {
            'E': 0, 'ENE': 22.5, 'NE': 45, 'NNE': 67.5,
            'N': 90, 'NNW': 112.5, 'NW': 135, 'WNW': 157.5,
            'W': 180, 'WSW': 202.5, 'SW': 225, 'SSW': 247.5,
            'S': 270, 'SSE': 292.5, 'SE': 315, 'ESE': 337.5,
        }

        # Aplicar la recategorización
        for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
            X[col] = X[col].map(mapeo_coord)
            X[f'{col}_rad'] = np.deg2rad(X[col])
            X[f'{col}_sin'] = np.sin(X[f'{col}_rad']).round(5)
            X[f'{col}_cos'] = np.cos(X[f'{col}_rad']).round(5)

        # Eliminar columnas originales y columnas radianes
        columns_to_drop = [f'{col}_rad' for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']] + ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        X = X.drop(columns=columns_to_drop, axis=1)

        return X


class YesNoRecat(BaseEstimator, TransformerMixin):
    """
    Recategoriza los valores seleccionados 
    de la lista de estados de lluvia del dia actual. 
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Mapear 'No' a 0 y 'Yes' a 1
        X['RainToday'] = X['RainToday'].map({'No': 0, 'Yes': 1}).astype(float)
        return X

# class ManualStandardScaler(BaseEstimator, TransformerMixin):
#     """
#     Normaliza los datos según Z-score con el scaler generado en train (desarrollo).
#     Se realiza de manera manual con los atributos del scaler.
#     """
#     def __init__(self, scaler_path='scaler_model_25.pkl'):
#         self.scaler_path = scaler_path
#         self.mean_ = None
#         self.scale_ = None

#     def fit(self, X, y=None):
#         # Cargar el scaler previamente ajustado
#         scaler = joblib.load(self.scaler_path)
#         self.mean_ = pd.Series(scaler.mean_, index=self.get_column_names(X))
#         self.scale_ = pd.Series(scaler.scale_, index=self.get_column_names(X))
#         return self

#     def transform(self, X):
#         # Convertir a DataFrame si es un array de NumPy
#         if isinstance(X, np.ndarray):
#             X = pd.DataFrame(X, columns=self.get_column_names(X))

#         # Asegurar que todas las columnas a escalar estén presentes en X
#         if not set(X.columns).issubset(set(self.mean_.index)):
#             raise ValueError("Columns to scale not found in input DataFrame.")

#         # Seleccionar solo las columnas numéricas
#         numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

#         # Calcular la normalización manualmente solo para columnas numéricas
#         X_scaled_numeric = (X[numeric_columns] - self.mean_[numeric_columns]) / self.scale_[numeric_columns]

#         # Mantener las columnas no numéricas sin cambios
#         X_scaled = pd.concat([X_scaled_numeric, X.drop(columns=numeric_columns)], axis=1)

#         return X_scaled

#     def get_column_names(self, X):
#         # Obtener los nombres de las columnas, ya sea desde un DataFrame o un array de NumPy
#         if isinstance(X, pd.DataFrame):
#             return X.columns
#         elif isinstance(X, np.ndarray):
#             return np.arange(X.shape[1])
#         else:
#             raise ValueError("Unsupported input type. Use DataFrame or NumPy array.")


class ManualStandardScaler(BaseEstimator, TransformerMixin):
    """
    Normaliza los datos según Z-score con el scaler generado en train (desarrollo).
    Se realiza de manera manual con los atributos del scaler.
    """
    def __init__(self, scaler_path='scaler_model_25.pkl'):
        self.scaler_path = scaler_path
        self.mean_ = None
        self.scale_ = None
        self.exclude_columns = ['RainToday']

    def fit(self, X, y=None):
        # Cargar el scaler previamente ajustado
        scaler = joblib.load(self.scaler_path)
        # Excluir las columnas especificadas del ajuste del scaler
        columns_to_scale = [col for col in X.columns if col not in self.exclude_columns]
        self.mean_ = scaler.mean_
        self.scale_ = scaler.scale_
        return self

    def transform(self, X):
        # Excluir las columnas especificadas de la transformación
        columns_to_scale = [col for col in X.columns if col not in self.exclude_columns]

        # Asegurar que las columnas a escalar estén presentes en X
        if not set(columns_to_scale).issubset(set(X.columns)):
            raise ValueError("Columns to scale not found in input DataFrame.")

        # Calcular la normalización manualmente
        X_scaled = (X[columns_to_scale] - self.mean_) / self.scale_

        # Mantener las columnas excluidas sin cambios
        X_scaled[self.exclude_columns] = X[self.exclude_columns]

        return X_scaled

# class ManualStandardScaler(BaseEstimator, TransformerMixin):
#     def __init__(self, scaler_path='no_date_scaler_model.pkl'):
#         self.scaler_path = scaler_path
#         self.mean_ = None
#         self.scale_ = None
#         self.exclude_columns = ['RainToday']

#     def fit(self, X, y=None):
#         # Cargar el scaler previamente ajustado
#         scaler = joblib.load(self.scaler_path)
#         # Excluir las columnas especificadas del ajuste del scaler
#         columns_to_scale = [col for col in X.columns if col not in self.exclude_columns]
#         self.mean_ = scaler.mean_
#         self.scale_ = scaler.scale_
#         return self

#     def transform(self, X):
#         # Excluir las columnas especificadas de la transformación
#         columns_to_scale = [col for col in X.columns if col not in self.exclude_columns]

#         # Asegurar que las columnas a escalar estén presentes en X
#         if not set(columns_to_scale).issubset(set(X.columns)):
#             raise ValueError("Columns to scale not found in input DataFrame.")

#         # Calcular la normalización manualmente
#         X_scaled = (X[columns_to_scale] - self.mean_) / self.scale_

#         # Mantener las columnas excluidas sin cambios
#         X_scaled[self.exclude_columns] = X[self.exclude_columns]

#         return X_scaled

# class ClassifierModel(BaseEstimator, TransformerMixin):
#     """
#     Implementa la Red Neuronal de clasificación. 
#     """
#     def __init__(self, model_path='nn_clf_model.h5'):
#         self.model_path = model_path
#         self.model = None

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         if self.model is None:
#             self.model = load_model(self.model_path)

#         # Realizar predicciones
#         clf_pred = self.model.predict(X)
#         return clf_pred

#     def predict(self, X):
#         return self.transform(X)
    
class ClassifierModel(BaseEstimator, TransformerMixin):
    """
    Implementa la Red Neuronal de clasificación. 
    """
    def __init__(self, model_path='nn_clf_model.h5'):
        self.model_path = model_path
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.model is None:
            self.model = load_model(self.model_path)

        # Realizar predicciones
        clf_pred = self.model.predict(X)
        return clf_pred

    def predict(self, X):
        return self.transform(X)


class RegressorModel(BaseEstimator, TransformerMixin):
    """
    Implementa la Red Neuronal de regresión. 
    """
    def __init__(self, model_path='nn_reg_model.h5'):
        self.model_path = model_path
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.model is None:
            self.model = load_model(self.model_path)

        # Realizar predicciones
        clf_pred = self.model.predict(X)
        return clf_pred

    def predict(self, X):
        return self.transform(X)


# ------------------- PIPELINES ----------------------
pipe_clf = Pipeline([
    ('date_extraction', DateExtraction()),
    ('coord_recat', CoordRecat()),
    ('yesno_recat', YesNoRecat()),
    ('manual_standard_scaler', ManualStandardScaler(scaler_path='scaler_model.pkl')),
    ('classifier_model', ClassifierModel(model_path='nn_clf_model.h5')),
])

pipe_reg = Pipeline([
    ('date_extraction', DateExtraction()),
    ('coord_recat', CoordRecat()),
    ('yesno_recat', YesNoRecat()),
    ('manual_standard_scaler', ManualStandardScaler(scaler_path='scaler_model.pkl')),
    ('regressor_model', RegressorModel(model_path='nn_reg_model.h5')),
])


# --------------- TEST -----------------------
# Definir los datos de entrada para testear
data = {
    'Date': ['2023-05-17'],
    'MinTemp': [15.00],
    'MaxTemp': [28.00],
    'Rainfall': [188.00],
    'Evaporation': [2.20],
    'Sunshine': [0.00],
    'WindGustSpeed': [150.00],
    'WindSpeed9am': [220.00],
    'WindSpeed3pm': [200.00],
    'Humidity9am': [68.00],
    'Humidity3pm': [90.00],
    'Pressure9am': [1014.00],
    'Pressure3pm': [1011.00],
    'Cloud9am': [8.00],
    'Cloud3pm': [8.00],
    'Temp9am': [15.00],
    'Temp3pm': [21.00],
    'WindGustDir': ['E'],
    'WindDir9am': ['NNE'],
    'WindDir3pm': ['E'],
    'RainToday': ['Yes']
    }