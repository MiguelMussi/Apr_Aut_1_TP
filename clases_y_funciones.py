# clases y funciones del pipeline de preprocesamiento.
import pandas as pd
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Lista de variables y sus tipos correspondientes
vars = [
    ('Date', 'date'),
    ('Temp9am', 'float'),
    ('Temp3pm', 'float'),
    ('MinTemp', 'float'),
    ('MaxTemp', 'float'),
    ('Humidity9am', 'float'),
    ('Humidity3pm', 'float'),
    ('Pressure9am', 'float'),
    ('Pressure3pm', 'float'),
    ('Cloud9am', 'float'),
    ('Cloud3pm', 'float'),
    ('Evaporation', 'float'),
    ('Sunshine', 'float'),
    ('WindGustDir', 'coord'),
    ('WindGustSpeed', 'float'),
    ('WindDir9am', 'coord'),
    ('WindSpeed9am', 'float'),
    ('WindDir3pm', 'coord'),
    ('WindSpeed3pm', 'float'),
    ('RainToday', 'state'),
    ('Rainfall', 'float')
]

def date_extraction(df):
    """
    Función que extrae el atributo "MES" de la fecha
    y lo recategoriza de manera cíclica con seno y coseno
    """
    # Asegurar que 'date' es una columna de tipo datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extrae el mes de la fecha
    # data['Year'] = data['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    # data['Day'] = data['Date'].dt.day

    # Codifica el mes en month_sin y month_cos
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12).round(5)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12).round(5)

    # Elimina la columna 'month'
    df = df.drop('Date', axis=1)

    return df

def coord_recat(df):
    """
    Función para recategorizar los valores seleccionados 
    de la lista de coordenadas. 
    """
    # Mapeo de las coordenadas a grados
    mapeo_coord = {
        'E': 0, 'ENE': 22.5, 'NE': 45, 'NNE': 67.5,
        'N': 90, 'NNW': 112.5, 'NW': 135, 'WNW': 157.5,
        'W': 180, 'WSW': 202.5, 'SW': 225, 'SSW': 247.5,
        'S': 270, 'SSE': 292.5, 'SE': 315, 'ESE': 337.5,
        }
    # Variables con direcciones de viento
    df['WindGustDir'] = df['WindGustDir'].map(mapeo_coord)
    df['WindDir9am'] = df['WindDir9am'].map(mapeo_coord)
    df['WindDir3pm'] = df['WindDir3pm'].map(mapeo_coord)

    # Conversión de grados a radianes
    df['WindGustDir_rad'] = np.deg2rad(df['WindGustDir'])
    df['WindDir9am_rad'] = np.deg2rad(df['WindDir9am'])
    df['WindDir3pm_rad'] = np.deg2rad(df['WindDir3pm'])

    # Codificación cíclica con senos y cosenos
    df['WindGustDir_sin'] = np.sin(df['WindGustDir_rad']).round(5)
    df['WindGustDir_cos'] = np.cos(df['WindGustDir_rad']).round(5)
    df['WindDir9am_sin'] = np.sin(df['WindDir9am_rad']).round(5)
    df['WindDir9am_cos'] = np.cos(df['WindDir9am_rad']).round(5)
    df['WindDir3pm_sin'] = np.sin(df['WindDir3pm_rad']).round(5)
    df['WindDir3pm_cos'] = np.cos(df['WindDir3pm_rad']).round(5)

    # Eliminación de las columnas originales y las columnas en radianes
    df = df.drop(['WindGustDir', 'WindDir9am', 'WindDir3pm', 'WindGustDir_rad', 'WindDir9am_rad', 'WindDir3pm_rad'], axis=1)
    
    return df

def yesno_recat(df):
    """
    Función para recategorizar los valores seleccionados 
    de la lista de estados de lluvia del dia actual. 
    """
    # Reemplazo de Yes/No
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
    df['RainToday'] = df['RainToday'].astype(int)

    return df
  
def standard_scaler_function(df):
    """
    Función para normalizar los datos según Z-score. 
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def clf_model(df):
    """
    Función para implementar la Red Neuronal de clasificación. 
    """
    # Carga de modelo
    clf_model = load_model('nn_cls_model.h5')

    # Predecir
    clf_pred = clf_model.predict(df)
    return clf_pred

def reg_model(df):
    """
    Función para implementar la Red Neuronal de regresión. 
    """
    # Carga de modelo
    reg_model = load_model('nn_reg_model.h5')

    # Predecir
    reg_pred = reg_model.predict(df)
    return reg_pred
  
class DateExtraction(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return date_extraction(X)
    
class CoordRecat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return coord_recat(X)

class YesNoRecat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return yesno_recat(X)

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return standard_scaler_function(X)

# class ClfModel(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.model = load_model('nn_clf_model.h5')

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return self.model.predict(X)
    
#     def predict(self, X):
#         return self.model.predict(X)

class ClfModel(BaseEstimator, TransformerMixin):
    def __init__(self, model_path='nn_clf_model.h5'):
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            if self.model is None:
                self.model = load_model(self.model_path)
            return self.model.predict(X)
        except Exception as e:
            print(f"Error al cargar el modelo de clasificación: {e}")
            return None
    
    def predict(self, X):
        return self.transform(X)

# class RegModel(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.model = load_model('nn_reg_model.h5')

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return self.model.predict(X)
    
#     def predict(self, X):
#         return self.model.predict(X)

class RegModel(BaseEstimator, TransformerMixin):
    def __init__(self, model_path='nn_reg_model.h5'):
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            if self.model is None:
                self.model = load_model(self.model_path)
            return self.model.predict(X)
        except Exception as e:
            print(f"Error al cargar el modelo de clasificación: {e}")
            return None
    
    def predict(self, X):
        return self.transform(X)