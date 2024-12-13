# Entry point for the Streamlit app and API deployment.

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit.components.v1 import html
from config import (
    MODEL_PATH,
    COMMUNE_DATA_PATH,
    BUILDING_CONDITIONS,
    EQUIPPED_KITCHEN,
    PROPERTY_SUBTYPES,
)
from preprocessing.cleaning_data import Preprocessor
from predict.prediction import Predictor

# Initialize preprocessor and predictor
preprocessor = Preprocessor()
predictor = Predictor(MODEL_PATH)

# Get list of communes
COMMUNE_DATA = pd.read_csv(COMMUNE_DATA_PATH)
COMMUNES = sorted(
    Preprocessor.get_unique_values(column="commune", import_path=COMMUNE_DATA_PATH)
)

# Set page configuration
st.set_page_config(page_title="ImmoEliza Price Prediction App", layout="wide")

# Streamlit app layout
st.title("ImmoEliza Price Prediction App")

# Create two columns: one for the map and one for the input fields ([]) sets ratio
col1, col2 = st.columns([2, 1])

# Right column: Input fields
with col2:
    st.subheader("Select Feature Values:")
    input_data = {
        "living_area": st.number_input("Living Area (m²)"),
        "commune": st.selectbox("Select a Commune", COMMUNES),
        "building_condition": st.select_slider(
            "Select the Building Condition:", BUILDING_CONDITIONS
        ),
        "subtype_of_property": st.selectbox(
            "Select a Property Subtype:", PROPERTY_SUBTYPES
        ),
        "equipped_kitchen": st.selectbox("Select Kitchen Equipment:", EQUIPPED_KITCHEN),
        "terrace": st.selectbox("Terrace", ["No", "Yes"]),
    }
    st.write("")

    # Prediction button
    if st.button("Predict Price"):
        try:
            # Preprocess input data
            preprocessed_data = preprocessor.preprocess(
                data=input_data, import_path=COMMUNE_DATA_PATH
            )  # Returns numpy array in original order
            # Get prediction
            prediction = predictor.predict(preprocessed_data)
            predicted_price = prediction.iloc[0]
            # Announce prediction
            st.success(f"Predicted Price: €{predicted_price:,.2f}")
        except ValueError as e:
            st.error(f"Error: {e}")

# Get coordinates of selected commune
selected_commune = input_data["commune"]
commune_coordinates = COMMUNE_DATA[COMMUNE_DATA["commune"] == selected_commune][
    ["latitude", "longitude"]
]

# Left column: Show map of selected commune using folium
with col1:
    if not commune_coordinates.empty:
        lat = commune_coordinates.iloc[0]["latitude"]
        lon = commune_coordinates.iloc[0]["longitude"]

        # Create a folium map centered around the commune coordinates with a zoom level
        m = folium.Map(
            location=[lat, lon], zoom_start=8
        )  # Adjust zoom level here (e.g., 12 for medium zoom)

        # Use map style with attribution

        folium.TileLayer(
            "CartoDB voyager",
            attr="Map tiles by CartoDB, under CC BY 3.0, Data by OpenStreetMap, under ODbL",
        ).add_to(m)

        # Add a marker for the commune
        folium.Marker([lat, lon], popup=f"{selected_commune}").add_to(m)

        # Display map in Streamlit
        map_html = m._repr_html_()  # Render the map as HTML
        st.components.v1.html(map_html, height=500)

    else:
        st.warning("No coordinates available for the selected commune.")


# TODO: Export list of features from model creation and import here (to make sure input features are in the same order as when model was trained)
