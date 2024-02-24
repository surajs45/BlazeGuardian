import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import confusion_matrix

st.title("_BlazeGuardian_")

st.write("""
## Explore the factors that contribute to the wildfires in Alberta and Beyond
This machine learning model not only displays to you how  much each factor
contributes to a wildfire, but also allows you to explore the data further by
segmenting it into various causes and possible factors to help mitigate the risk
of wildfires particularly in first nations communities. Please download the
dataset below for your reference.
""")

st.download_button("Alberta Wildfires Data Set", "/content/fp-historical-wildfire-data-2006-2021.csv", file_name = "fp-historical-wildfire-data-2006-2021.csv")

st.subheader("Access to Detailed Report")
st.write("Use the button below to acces a in-depth analysis of our findings.")
st.link_button("Access Report", "https://docs.google.com/document/d/1frUK1DXamQg05Hj76eCshZuxbO8vuOiJT_kCOUjOk_4/edit?usp=sharing")

st.sidebar.title("_Explore the dataset!_")


st.title("_BlazeGuardian_")

st.write("""
## Explore the factors that contribute to the wildfires in Alberta and Beyond
The input variables provided to the prediction function represent key characteristics of a fire incident, including its geographical location (latitude and longitude), potential causes such as recreation or lightning strikes, fire spread rate, fuel types involved, and activity classes related to the fire. The XGBoost model leverages these features to predict a severity index value, indicating the anticipated severity or extent of the fire incident. This index value serves as a quantitative or qualitative measure of the fire's severity, aiding in decision-making and resource allocation for firefighting and emergency response efforts.
""")

df1 = pd.read_csv("Wildfire_Preprocessed_Data.csv")

st.dataframe(df1)

import xgboost as xgb
import numpy as np
import pickle
import matplotlib.pyplot as plt

loaded_model = xgb.XGBClassifier()
loaded_model.load_model("model_with_coordinates.json")

required_columns = ['fire_location_latitude', 'fire_location_longitude', 'general_cause_desc_Recreation',
                    'fire_spread_rate', 'fuel_type_C2', 'fuel_type_O1a', 'activity_class_Cooking and Warming',
                    'general_cause_desc_Lightning']


X_example = df1[required_columns].copy()

example_pred = loaded_model.predict(X_example)


# Define the prediction function
def predict_fire_size(latitude, longitude, general_cause_desc_Recreation, fire_spread_rate,
                      fuel_type_C2, fuel_type_O1a, activity_class_Cooking, general_cause_desc_Lightning):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'fire_location_latitude': [latitude],
        'fire_location_longitude': [longitude],
        'general_cause_desc_Recreation': [general_cause_desc_Recreation],
        'fire_spread_rate': [fire_spread_rate],
        'fuel_type_C2': [fuel_type_C2],
        'fuel_type_O1a': [fuel_type_O1a],
        'activity_class_Cooking and Warming': [activity_class_Cooking],
        'general_cause_desc_Lightning': [general_cause_desc_Lightning]
    })

    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data)

    # Return the prediction
    return prediction[0]

# Streamlit UI
st.title('Fire Size Prediction')

# Input fields for prediction
latitude = st.number_input('Latitude')
longitude = st.number_input('Longitude')
general_cause_desc_Recreation = st.selectbox('General Cause Recreation', [0, 1])
fire_spread_rate = st.number_input('Fire Spread Rate')
fuel_type_C2 = st.selectbox('Fuel Type C2', [0, 1])
fuel_type_O1a = st.selectbox('Fuel Type O1a', [0, 1])
activity_class_Cooking = st.selectbox('Activity Class Cooking and Warming', [0, 1])
general_cause_desc_Lightning = st.selectbox('General Cause Lightning', [0, 1])

# Button to trigger prediction
if st.button('Predict Fire Size'):
    # Call the prediction function
    prediction = predict_fire_size(latitude, longitude, general_cause_desc_Recreation, fire_spread_rate,
                                   fuel_type_C2, fuel_type_O1a, activity_class_Cooking, general_cause_desc_Lightning)

    # Display the prediction
    st.subheader(f'Predicted Fire Size: {prediction}')

    longitudes = range(0, 120)
    latitudes = range(0, 120)
    predictions = [predict_fire_size(latitude, longitude, general_cause_desc_Recreation, fire_spread_rate,
                                     fuel_type_C2, fuel_type_O1a, activity_class_Cooking,
                                     general_cause_desc_Lightning) for latitude in latitudes]

    # Plot the bar graph
    fig, ax = plt.subplots()
    ax.bar(latitudes, predictions)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Predicted Fire Size')
    ax.set_title('Predicted Fire Size vs Latitude')
    st.pyplot(fig)

