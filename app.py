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
from typing import List, Any


def main():
    # Set page configuration
    st.set_page_config(
        page_title="ImmoEliza Real-Estate Price Predictor", layout="wide"
    )

    # Load data and initialize app
    app = StreamlitApp(
        MODEL_PATH, COMMUNE_DATA_PATH, col_pair1=[1, 15], col_pair2=[1.2, 2]
    )
    # Set layout
    app.set_layout()
    # Set input mask
    input_data = app.input_features()
    selected_commune = input_data.get("commune")
    if selected_commune:
        app.display_map(selected_commune)


class StreamlitApp:
    """Class to deploy a streamlit app for property price predictions using a trained machine learning model."""

    def __init__(
        self,
        model_path: str,
        location_data_path: str,
        col_pair1: List[Any],
        col_pair2: List[Any],
    ):
        """Initializes the application with necessary components and data paths."""
        self.model_path = model_path
        self.location_data_path = location_data_path

        # Initialize preprocessor and predictor
        self.preprocessor = Preprocessor()
        self.predictor = Predictor(self.model_path)

        # Load commune data
        self.commune_data = pd.read_csv(self.location_data_path)
        self.communes = sorted(
            Preprocessor.get_unique_values(
                column="commune", import_path=self.location_data_path
            )
        )

        # Initialize columns
        self.col1, self.col2, self.col3, self.col4 = self.create_columns(
            col_pair1, col_pair2
        )

    def create_columns(self, col_pair1: List[Any], col_pair2: List[Any]):
        """Creates the layout columns for the app."""
        col1, col2 = st.columns(col_pair1)
        # Set divider
        st.markdown("---")
        col3, col4 = st.columns(col_pair2)
        return col1, col2, col3, col4

    def set_layout(self):
        """Sets up the layout of the Streamlit app."""
        with self.col1:
            # Set logo
            st.write("")  # To move logo down
            st.image("./images/logo.png", width=90)

        with self.col2:
            # Set title
            st.title("ImmoEliza Real-Estate Price Predictor")

    def input_features(self):
        """Displays input fields for user to enter feature values."""

        with self.col3:
            st.subheader("Select Feature Values:")
            input_data = {
                "living_area": st.number_input("Living Area (m²)"),
                "commune": st.selectbox("Select a Commune", self.communes),
                "building_condition": st.select_slider(
                    "Select the Building Condition:", BUILDING_CONDITIONS
                ),
                "subtype_of_property": st.selectbox(
                    "Select a Property Subtype:", PROPERTY_SUBTYPES
                ),
                "equipped_kitchen": st.selectbox(
                    "Select Kitchen Equipment:", EQUIPPED_KITCHEN
                ),
                "terrace": st.selectbox("Terrace", ["No", "Yes"]),
            }
            st.write("")

            # Prediction button
            if st.button("Predict Price"):
                self.predict_price(input_data)

        return input_data

    def predict_price(self, input_data: dict):
        """Handles the prediction process and displays results."""
        try:
            # Preprocess input data
            preprocessed_data = self.preprocessor.preprocess(
                data=input_data, import_path=self.location_data_path
            )
            # Get prediction
            prediction = self.predictor.predict(preprocessed_data)
            predicted_price = prediction.iloc[0]
            # Announce prediction
            st.success(f"Predicted Price: €{predicted_price:,.2f}")
        except ValueError as e:
            st.error(f"Error: {e}")

    def display_map(self, selected_commune: str):
        """Displays a map with the location of the selected commune."""
        commune_coordinates = self.commune_data[
            self.commune_data["commune"] == selected_commune
        ][["latitude", "longitude"]]

        with self.col4:
            if not commune_coordinates.empty:
                lat = commune_coordinates.iloc[0]["latitude"]
                lon = commune_coordinates.iloc[0]["longitude"]

                # Create a folium map centered around the commune coordinates
                m = folium.Map(location=[lat, lon], zoom_start=8)
                folium.TileLayer(
                    "CartoDB voyager",
                    attr="Map tiles by CartoDB, under CC BY 3.0, Data by OpenStreetMap, under ODbL",
                ).add_to(m)

                # Add a marker for the commune
                folium.Marker([lat, lon], popup=f"{selected_commune}").add_to(m)

                # Display map in Streamlit
                map_html = m._repr_html_()
                st.components.v1.html(map_html, height=500)
            else:
                st.warning("No coordinates available for the selected commune.")


# Initialize and run the app
if __name__ == "__main__":
    main()


# TODO: Export list of features from model creation and import here (to make sure input features are in the same order as when model was trained)
