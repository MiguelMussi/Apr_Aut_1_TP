# app.py
import streamlit as st
from sklearn.pipeline import Pipeline
import joblib
from keras.models import load_model
import pandas as pd
import os
from datetime import datetime
from clases_y_funciones import vars, DateExtraction, CoordRecat, YesNoRecat, StandardScalerTransformer, ClfModel, RegModel

path_dir = os.path.dirname(os.path.abspath(__file__))
# Se carga el pipeline del modelo de clasificación.
pkl_clf_path = os.path.join(path_dir, 'rain_clf_pipeline.joblib')
pipe_clf = joblib.load(pkl_clf_path)

# Se carga el pipeline del modelo de regresión.
pkl_reg_path = os.path.join(path_dir, 'rain_reg_pipeline.joblib')
pipe_reg = joblib.load(pkl_reg_path)

st.title('Rain Predictor Model')


def get_user_input():
    """
    esta función genera los inputs del frontend de streamlit para que el usuario pueda cargar los valores.
    Además, contiene el botón para hacer el submit y obtener la predicción.
    No hace falta hacerlo así, las posibilidades son infinitas.
    """
    input_dict = {}

    with st.form(key='my_form'):

        for var, data_type in vars:
            if data_type == 'date':
                input_dict[var] = st.date_input(f"{var} Input", value=datetime.now())
            elif data_type == 'float':
                input_dict[var] = st.number_input(f"{var} Input", value=0.0, step=0.01)
            elif data_type == 'coord':
                coords = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW','W', 'WSW', 'SW', 'SSW','S', 'SSE', 'SE', 'ESE']
                input_dict[var] = st.selectbox(f"{var} Input", coords)
            elif data_type == 'state':
                states = ['Yes', 'No']
                input_dict[var] = st.selectbox(f"{var} Input", states)
        
     
        submit_button = st.form_submit_button(label='Predict')

    return pd.DataFrame([input_dict]), submit_button


user_input, submit_button = get_user_input()


# When the 'Submit' button is pressed, perform the prediction
if submit_button:
    # Predict rain status and value
    clf_prediction = pipe_clf.predict(user_input)
    reg_prediction = pipe_reg.predict(user_input)

    # Display the prediction
    st.header("Predicción de estado de lluvia")
    st.write(clf_prediction)
    st.header("Predicción de cantidad de lluvia")
    st.write(reg_prediction)
    

st.markdown(
    """
    Modelo de predicción de lluvias creado por:<br>
    [ Brisa Menescaldi | Miguel Mussi ](https://github.com/MiguelMussi/Apr_Aut_1_TP)
    """, unsafe_allow_html=True
)