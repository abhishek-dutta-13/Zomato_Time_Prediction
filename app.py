import streamlit as st
import pandas as pd
import numpy as np
import pickle
from src.utils import *
from src.logger import logger
import json




model = load_obj(os.path.join("artifacts", "model.pkl"))
preprocessor = load_obj(os.path.join("artifacts", "preprocessor_obj.pkl"))

# Read categorical_column_mapping.json file
with open(os.path.join("categorical_column_mapping.json"), "r") as f:
    categorical_column_mapping = json.load(f)


####################### Streamlit ############################################

# Setting the page title and layout
st.set_page_config(page_title="Zomato Time Prediction", layout="wide")

# Adding a header and subheader with some styling
st.title("Zomato Delivery Time Prediction")
st.markdown("""
This application predicts the delivery time based on various factors such as delivery person's details, weather conditions, vehicle condition, and more.
""")

col1, col2 = st.columns(2)

with col1:
    Delivery_person_Age = st.number_input("Age")
    Delivery_person_Ratings = st.number_input("Ratings")
    Weather_condtitions = st.selectbox("Weather Condtions: ", categorical_column_mapping["Weather_conditions"])
    Road_traffic_density = st.selectbox("Traffic: ", categorical_column_mapping["Road_traffic_density"])
    Vehicle_condition = st.selectbox("Vehicle_condition: ", [2, 1, 0, 3])
with col2:
    Type_of_vehicle  = st.selectbox("Type of vechicle: ", categorical_column_mapping["Type_of_vehicle"])
    Type_of_order = st.selectbox("Type_of_order: ", categorical_column_mapping["Type_of_order"])
    multiple_diliveries = st.selectbox("Multiple_diliveries", [3., 1., 0., 2.])
    Festival = st.radio("Festival: ", categorical_column_mapping["Festival"])
    City = st.selectbox("City: ", categorical_column_mapping["City"])
    Distance = st.number_input("Distance")

data = {
    "Delivery_person_Age": Delivery_person_Age,
    "Delivery_person_Ratings": Delivery_person_Ratings,
    "Weather_conditions": Weather_condtitions,
    "Road_traffic_density": Road_traffic_density,
    "Vehicle_condition": Vehicle_condition,
    "Type_of_vehicle": Type_of_vehicle,
    "Type_of_order": Type_of_order,
    "multiple_deliveries": multiple_diliveries,
    "Festival": Festival,
    "City": City,
    "distance": Distance # Assuming distance is in km or miles
}
data_df = pd.DataFrame(data=data, index = [0])

# Prediction button
submit = st.sidebar.button("Predict Delivery Time")

# Displaying the prediction
if submit:
    try:
        new_data = preprocessor.transform(data_df)
        logger.info("transformation done for new data!!!")
        prediction = model.predict(new_data)
        logger.info("prediction done!!!")
        st.write(f"The predicted delivery time is {prediction[0]} minutes.")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
