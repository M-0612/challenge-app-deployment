# Entry point for the Streamlit app and API deployment.

import streamlit as st
import pandas as pd
import folium
from preprocessing.cleaning_data import Preprocessor
from predict.prediction import Predictor
from typing import Any, Dict, List
from config import (
    MODEL_PATH,
    COMMUNE_DATA_PATH,
    BUILDING_CONDITIONS,
    EQUIPPED_KITCHEN,
    PROPERTY_SUBTYPES,
)


def main():
    """Main entry point for the Streamlit app, sets up the page, initializes the app, and handles user interactions."""
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
    """Class to deploy a Streamlit app for property price predictions using a trained machine learning model."""

    def __init__(
        self,
        model_path: str,
        location_data_path: str,
        col_pair1: List[Any],
        col_pair2: List[Any],
    ) -> None:
        """Initializes the application with necessary components and data paths.

        Args:
            model_path (str): Path to the trained machine learning model.
            location_data_path (str): Path to the CSV file containing commune data.
            col_pair1 (List[Any]): Layout configuration for the first pair of columns.
            col_pair2 (List[Any]): Layout configuration for the second pair of columns.

        Returns:
            None: Initializes the application.
        """
        self.model_path = model_path
        self.location_data_path = location_data_path

        # Initialize preprocessor and predictor
        self.preprocessor = Preprocessor()
        self.predictor = Predictor(self.model_path)

        # Load location data (commune names, latitude, longitude, distance to nearest large city)
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

    def create_columns(self, col_pair1: List[Any], col_pair2: List[Any]) -> tuple:
        """Creates the layout columns for the app.

        Args:
            col_pair1 (List[Any]): Layout configuration for the first pair of columns.
            col_pair2 (List[Any]): Layout configuration for the second pair of columns.

        Returns:
            tuple: A tuple containing the four column layout.
        """
        col1, col2 = st.columns(col_pair1)
        # Set divider
        st.markdown("---")
        col3, col4 = st.columns(col_pair2)
        return col1, col2, col3, col4

    def set_layout(self) -> None:
        """Sets up the layout of the Streamlit app."""
        with self.col1:
            # Set logo
            st.write("")  # To move logo down
            st.image("./images/logo.png", width=90)

        with self.col2:
            # Set title
            st.title("ImmoEliza Real-Estate Price Predictor")

    def input_features(self) -> Dict[str, Any]:
        """Displays input fields for user to enter feature values.

        Returns:
            Dict[str, Any]: A dictionary containing the input data from the user.
        """
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

    def predict_price(self, input_data: Dict[str, Any]) -> None:
        """Handles the prediction process and displays results.

        Args:
            input_data (Dict[str, Any]): A dictionary containing the input data for prediction.
        """
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

    def display_map(self, selected_commune: str) -> None:
        """Displays a map with the location of the selected commune.

        Args:
            selected_commune (str): The name of the commune to display on the map.
        """
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
