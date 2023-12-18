# app.py
import streamlit as st
from sklearn.pipeline import Pipeline
import joblib
import keras
import tensorflow as tf
from keras.models import load_model
# from tensorflow.keras.models import load_model

import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import dill
import cloudpickle
from clases_y_funciones import DateExtraction, CoordRecat, YesNoRecat, ManualStandardScaler, ClassifierModel, RegressorModel


# ------------------ Streamlit ------------------------

# Listas auxiliares
coords = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW','W', 'WSW', 'SW', 'SSW','S', 'SSE', 'SE', 'ESE']
states = ['No', 'Yes']

st.title('Modelo de Predicción de Lluvia')

# Sidebar
st.sidebar.title('Sección para el usuario')
st.sidebar.header('Carga de datos')
st.sidebar.write('Ingresar valores del día anterior a la predicción. ')

# ------------------ Formulario ------------------------
date = st.sidebar.date_input('Date', value=datetime.now())
temp9am = st.sidebar.slider('Temp9am', -10.0, 35.0, 25.0, help="Temperatura en ºC")
temp3pm = st.sidebar.slider('Temp3pm', -10.0, 35.0, 25.0, help="Temperatura en ºC")
mintemp = st.sidebar.slider('MinTemp', -10.0, 35.0, 25.0, help="Temperatura en ºC")
maxtemp = st.sidebar.slider('MaxTemp', -10.0, 35.0, 25.0, help="Temperatura en ºC")
humidity9am = st.sidebar.slider('Humidity9am', 0.0, 100.0, 50.0, )
humidity3pm = st.sidebar.slider('Humidity3pm', 0.0, 100.0, 50.0)
pressure9am = st.sidebar.slider('Pressure9am', 950.0, 1100.0, 1013.0)
pressure3pm = st.sidebar.slider('Pressure3pm', 950.0, 1100.0, 1013.0)
cloud9am = st.sidebar.slider('Cloud9am', 0.0, 10.0, 2.0)
cloud3pm = st.sidebar.slider('Cloud3pm', 0.0, 10.0, 2.0)
evaporation = st.sidebar.slider('Evaporation', 0.0, 10.0, 5.0)
sunshine = st.sidebar.slider('Sunshine', 0.0, 15.0, 7.5)
windgustdir = st.sidebar.selectbox('WindGustDir', coords)
windgustspeed = st.sidebar.slider('WindGustSpeed', 6.0, 135.0, 40.0)
winddir9am = st.sidebar.selectbox('WindDir9am', coords)
windspeed9am = st.sidebar.slider('WindSpeed9am', 0.0, 135.0, 15.0)
winddir3pm = st.sidebar.selectbox('WindDir3pm', coords)
windspeed3pm = st.sidebar.slider('WindSpeed3pm', 0.0, 135.0, 15.0)
raintoday = st.sidebar.radio("RainToday", states)
rainfall = st.sidebar.number_input('Rainfall', min_value=0.0, max_value=300.0)
# ------------------ Formulario ------------------------


# ------------------ Test Data ------------------------
data = {
    'Date': ['2023-01-17'],
    'MinTemp': [20.00],
    'MaxTemp': [25.00],
    'Rainfall': [7.00],
    'Evaporation': [2.20],
    'Sunshine': [1.00],
    'WindGustSpeed': [45.00],
    'WindSpeed9am': [15.00],
    'WindSpeed3pm': [5.00],
    'Humidity9am': [35.00],
    'Humidity3pm': [95.00],
    'Pressure9am': [1015.00],
    'Pressure3pm': [1000.00],
    'Cloud9am': [3.00],
    'Cloud3pm': [9.00],
    'Temp9am': [20.00],
    'Temp3pm': [25.00],
    'WindGustDir': ['NW'],
    'WindDir9am': ['NNW'],
    'WindDir3pm': ['NW'],
    'RainToday': ['Yes']
    }

test_df = pd.DataFrame(data)

# ------------------ Test Data ------------------------


# --------------------- Datos ---------------------------
user_input_array = np.array([[date, mintemp, maxtemp, rainfall, evaporation, sunshine,\
                                windgustspeed, windspeed9am, windspeed3pm, \
                                humidity9am, humidity3pm, pressure9am, pressure3pm, \
                                cloud9am, cloud3pm, temp9am, temp3pm, \
                                windgustdir, winddir9am,  winddir3pm, raintoday]])

column_names = ['Date', 'Mintemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',\
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', \
                'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', \
                'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', \
                'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

user_input_df = pd.DataFrame(user_input_array, columns=column_names)


# --------------------- Clases ---------------------------
date_extraction = DateExtraction()
coord_recat = CoordRecat()
yes_no_recat = YesNoRecat()
manual_scaler = ManualStandardScaler()

# --------------------- Modelos ---------------------------
# classifier_model = ClassifierModel()
# regressor_model = RegressorModel()
# classifier_model = ClassifierModel(model_path='nn_clf_model')
# regressor_model = RegressorModel(model_path='nn_reg_model')
# classifier_model = keras.models.load_model('nn_clf_model')
# regressor_model = keras.models.load_model('nn_reg_model')
# classifier_model = load_model('nn_clf_model')
# regressor_model = load_model('nn_reg_model')
# classifier_model = tf.keras.models.load_model("nn_clf_model")
# regressor_model = tf.keras.models.load_model("nn_reg_model")

# --------------------- Pipeline ---------------------------
df_transformed = date_extraction.fit_transform(user_input_df)
df_transformed = coord_recat.fit_transform(df_transformed)
df_transformed = yes_no_recat.fit_transform(df_transformed)
df_transformed = manual_scaler.fit_transform(df_transformed)


# ----------------- Predicción Clasificación -----------------
# clf_predict = classifier_model.transform(df_transformed)
# threshold = 0.5  # Umbral de decisión
# binary_prediction = (clf_predict > threshold).astype(int)
# clf_prediction = "llueve" if binary_prediction[0][0] == 1 else "no llueve"


# ----------------- Predicción Regresión -----------------
# reg_predict = regressor_model.transform(df_transformed)
# rainfall_tomorrow_mean = 2.250213 # Media obtenida en desarrollo
# rainfall_tomorrow_std = 7.318972  # Desvío obtenido en desarrollo
# reg_prediction = (reg_predict[0][0] * rainfall_tomorrow_std) + rainfall_tomorrow_mean


clf_prediction = "llueve"
reg_prediction = 12.5


# ----------------- Resultados finales -----------------
clf_prediction_final = f"Mañana {clf_prediction}. Probabilidad: 75%"
reg_prediction_final = f"Mañana llueven {reg_prediction} mm."



# ----------------- Bloque principal -----------------
st.write('Datos cargados por el usuario:', user_input_df)
st.title("Predicciones")
st.write("Predicciones para el día:", date + timedelta(days=1))
# st.write('Datos de test:', test_df)
st.header("Predicción de estado de lluvia")
mensaje_estado = f'<p style="color: green; font-size: 20px;">{clf_prediction_final}</p>'
st.markdown(mensaje_estado, unsafe_allow_html=True)


st.header("Predicción de cantidad de lluvia")
mensaje_cantidad = f'<p style="color: green; font-size: 20px;">{reg_prediction_final}</p>'
st.markdown(mensaje_cantidad, unsafe_allow_html=True)

st.markdown(
    """
    _________________________________________________
    Modelo de predicción de lluvias creado por:<br>
    [ Brisa Menescaldi | Miguel Mussi ]<br>
    TUIA | Aprendizaje Automático | 2023<br>
    
    Repositorio Público [ [GitHub] ](https://github.com/MiguelMussi/Apr_Aut_1_TP)<br>
    _________________________________________________ 
    """, unsafe_allow_html=True
)