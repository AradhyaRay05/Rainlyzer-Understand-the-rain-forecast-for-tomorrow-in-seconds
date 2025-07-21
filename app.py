import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Load the saved model and scaler
model = joblib.load("best_catboost_model.pkl")


dummy_data = {
    'Location': [0], 'MinTemp': [0.0], 'MaxTemp': [0.0], 'Rainfall': [0.0], 'Evaporation': [0.0], 'Sunshine': [0.0],
    'WindGustDir': [0], 'WindGustSpeed': [0.0], 'WindDir9am': [0], 'WindDir3pm': [0], 'WindSpeed9am': [0.0],
    'WindSpeed3pm': [0.0], 'Humidity9am': [0.0], 'Humidity3pm': [0.0], 'Pressure9am': [0.0], 'Pressure3pm': [0.0],
    'Cloud9am': [0.0], 'Cloud3pm': [0.0], 'Temp9am': [0.0], 'Temp3pm': [0.0], 'RainToday': [0],
    'Date_month': [1], 'Date_day': [1]
}
dummy_df = pd.DataFrame(dummy_data)
scaler = MinMaxScaler()
# Fit the scaler on dummy data with the correct column order
scaler.fit(dummy_df)


st.title('Rainfall Prediction')

# Define the mapping dictionaries (should be the same as used in training)
location = {'Portland':1, 'Cairns':2, 'Walpole':3, 'Dartmoor':4, 'MountGambier':5,
       'NorfolkIsland':6, 'Albany':7, 'Witchcliffe':8, 'CoffsHarbour':9, 'Sydney':10,
       'Darwin':11, 'MountGinini':12, 'NorahHead':13, 'Ballarat':14, 'GoldCoast':15,
       'SydneyAirport':16, 'Hobart':17, 'Watsonia':18, 'Newcastle':19, 'Wollongong':20,
       'Brisbane':21, 'Williamtown':22, 'Launceston':23, 'Adelaide':24, 'MelbourneAirport':25,
       'Perth':26, 'Sale':27, 'Melbourne':28, 'Canberra':29, 'Albury':30, 'Penrith':31,
       'Nuriootpa':32, 'BadgerysCreek':33, 'Tuggeranong':34, 'PerthAirport':35, 'Bendigo':36,
       'Richmond':37, 'WaggaWagga':38, 'Townsville':39, 'PearceRAAF':40, 'SalmonGums':41,
       'Moree':42, 'Cobar':43, 'Mildura':44, 'Katherine':45, 'AliceSprings':46, 'Nhil':47,
       'Woomera':48, 'Uluru':49}
windgustdir = {'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10,
       'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15}
winddir9am = {'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10,
       'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15}
winddir3pm = {'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10,
       'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15}


# Create input widgets for features
st.header("Enter Weather Data:")

# Numerical Features (using st.number_input)
location_input = st.selectbox('Location', options=list(location.keys()))
min_temp = st.number_input('MinTemp', value=0.0)
max_temp = st.number_input('MaxTemp', value=0.0)
rainfall = st.number_input('Rainfall', value=0.0)
evaporation = st.number_input('Evaporation', value=0.0)
sunshine = st.number_input('Sunshine', value=0.0)
wind_gust_speed = st.number_input('WindGustSpeed', value=0.0)
wind_speed9am = st.number_input('WindSpeed9am', value=0.0)
wind_speed3pm = st.number_input('WindSpeed3pm', value=0.0)
humidity9am = st.number_input('Humidity9am', value=0.0)
humidity3pm = st.number_input('Humidity3pm', value=0.0)
pressure9am = st.number_input('Pressure9am', value=0.0)
pressure3pm = st.number_input('Pressure3pm', value=0.0)
cloud9am = st.number_input('Cloud9am', value=0.0)
cloud3pm = st.number_input('Cloud3pm', value=0.0)
temp9am = st.number_input('Temp9am', value=0.0)
temp3pm = st.number_input('Temp3pm', value=0.0)
date_month = st.number_input('Date_month', value=1, min_value=1, max_value=12)
date_day = st.number_input('Date_day', value=1, min_value=1, max_value=31)


# Categorical Features (using st.selectbox)
wind_gust_dir_input = st.selectbox('WindGustDir', options=list(windgustdir.keys()))
wind_dir9am_input = st.selectbox('WindDir9am', options=list(winddir9am.keys()))
wind_dir3pm_input = st.selectbox('WindDir3pm', options=list(winddir3pm.keys()))
rain_today_input = st.selectbox('RainToday', options=['No', 'Yes'])

# Add a predict button
predict_button = st.button('Predict Rain Tomorrow')

if predict_button:
    # Create a dictionary with user input
    user_data = {
        'Location': location_input,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustDir': wind_gust_dir_input,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir9am_input,
        'WindDir3pm': wind_dir3pm_input,
        'WindSpeed9am': wind_speed9am,
        'WindSpeed3pm': wind_speed3pm,
        'Humidity9am': humidity9am,
        'Humidity3pm': humidity3pm,
        'Pressure9am': pressure9am,
        'Pressure3pm': pressure3pm,
        'Cloud9am': cloud9am,
        'Cloud3pm': cloud3pm,
        'Temp9am': temp9am,
        'Temp3pm': temp3pm,
        'RainToday': rain_today_input,
        'Date_month': date_month,
        'Date_day': date_day
    }

    # Convert categorical features to numerical using the mapping dictionaries
    user_data['Location'] = location[user_data['Location']]
    user_data['WindGustDir'] = windgustdir[user_data['WindGustDir']]
    user_data['WindDir9am'] = winddir9am[user_data['WindDir9am']]
    user_data['WindDir3pm'] = winddir3pm[user_data['WindDir3pm']]
    user_data['RainToday'] = 1 if user_data['RainToday'] == 'Yes' else 0

    # Create a pandas DataFrame from the user input
    user_df = pd.DataFrame([user_data])

    # Apply the same scaling as the training data
    user_X_scaled = scaler.transform(user_df)

    # Make prediction
    prediction = model.predict(user_X_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.write("Prediction: It is likely to rain tomorrow.")
    else:
        st.write("Prediction: It is unlikely to rain tomorrow.")
