# API code

"""
This will be the entry point for the Streamlit app and API deployment. It will initialize your app and handle user input, triggering data preprocessing and model prediction.

You must be able to manually add features values on the streamlit app in order to make predictions. 
Imagine you want to buy a house and you wonder if its price is not exaggerated. 

You can use this tool to simulate the value of the property to get an idea of ​​the market."""

import streamlit as st
import pandas as pd
from config import MODEL_PATH, COMMUNE_DATA_PATH, BUILDING_CONDITIONS, EQUIPPED_KITCHEN, PROPERTY_SUBTYPES
from preprocessing.cleaning_data import Preprocessor
#from predict.prediction import Predictor

# Initialize preprocessor and predictor
preprocessor = Preprocessor()
#predictor = Predictor(MODEL_PATH)

# Get list of communes
COMMUNES = Preprocessor.get_unique_values(column='commune', import_path=COMMUNE_DATA_PATH)

# Streamlit app layout
st.title("ImmoEliza Price Prediction App")

# Input fields
input_data = {
    'living_area': st.number_input("Living Area (m²)"),
    'commune': st.selectbox("Select a Commune", COMMUNES), #TODO: Communes have to be replaced and removed
    'building_condition': st.select_slider("Select the Building Condition:", BUILDING_CONDITIONS),
    'subtype_of_property': st.selectbox("Select a Property Subtype:", PROPERTY_SUBTYPES),
    'equipped_kitchen': st.selectbox("Select Kitchen Equipment:", EQUIPPED_KITCHEN),
    'terrace': st.selectbox("Terrace", ['No', 'Yes']) # TODO: Should be yes/no
}

# Prediction button
if st.button("Predict Price"):
    try:
        # Preprocess input data
        preprocessed_data = preprocessor.preprocess(data=input_data, import_path=COMMUNE_DATA_PATH)
        # Get prediction
        #predicted_price = predictor(preprocessed_data)
        # Announce prediction
        #st.success(f"Predicted Price: €{predicted_price:,.2f}")
    except ValueError as e:
        st.error(f"Error: {e}")