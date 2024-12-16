# Entry point for the Streamlit app and API deployment.

import streamlit as st
import pandas as pd
import folium
from branca.colormap import LinearColormap
from folium.plugins import HeatMap
from preprocessing.cleaning_data import Preprocessor
from predict.prediction import Predictor
from typing import Any, Dict, List
from config import (
    MODEL_PATH,
    DATA_PATH,
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
    app = StreamlitApp(MODEL_PATH, DATA_PATH, col_pair1=[1, 15], col_pair2=[1.2, 1.5])
    # Set app layout
    app.set_layout()
    # Set input mask and handle prediction button
    input_data = app.input_features()
    # Get the value of the selected commune
    selected_commune = input_data.get("commune")
    if selected_commune:
        # Display map of selected commune
        app.display_map(selected_commune)


class StreamlitApp:
    """Class to deploy a Streamlit app for property price predictions using a trained machine learning model."""

    def __init__(
        self,
        model_path: str,
        data_path: str,
        col_pair1: List[Any],
        col_pair2: List[Any],
    ) -> None:
        """Initializes the application with necessary components and data paths.

        Args:
            model_path (str): Path to the trained machine learning model.
            data_path (str): Path to the CSV file containing commune data.
            col_pair1 (List[Any]): Layout configuration for the first pair of columns.
            col_pair2 (List[Any]): Layout configuration for the second pair of columns.

        Returns:
            None: Initializes the application.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.col_pair1 = col_pair1
        self.col_pair2 = col_pair2

        # Initialize preprocessor and predictor
        self.preprocessor = Preprocessor()
        self.predictor = Predictor(self.model_path)

        # Load location data (commune names, latitude, longitude, distance to nearest large city)
        self.data = pd.read_csv(self.data_path)
        self.communes = sorted(
            Preprocessor.get_unique_values(data=self.data, column="commune")
        )

        # Initialize columns
        self.col1, self.col2, self.col3, self.col4 = self.create_columns(
            self.col_pair1, self.col_pair2
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
            st.title("ImmoEliza Property Price Predictor")

    def sync_fields(self, field):
        """Synchronize two fields dynamically. Here: commune and zip code."""
        if field == "zip_code":
            try:
                zip_code = int(st.session_state["zip_code"])
                matched_commune = self.data[self.data["zip_code"] == zip_code]
                if not matched_commune.empty:
                    st.session_state["commune"] = matched_commune.iloc[0]["commune"]
                else:
                    st.warning(f"No match found for zip code '{zip_code}'.")
            except ValueError:
                st.warning("Invalid zip code format. Please enter numbers only.")

        elif field == "commune":
            commune = st.session_state["commune"]
            matched_zip = self.data[self.data["commune"] == commune]
            if not matched_zip.empty:
                st.session_state["zip_code"] = str(matched_zip.iloc[0]["zip_code"])
            else:
                st.warning(f"No match found for commune '{commune}'.")

    def input_features(self) -> Dict[str, Any]:
        """Displays input fields for user to enter feature values.

        Returns:
            Dict[str, Any]: A dictionary containing the input data from the user.
        """
        # Initialize session state for commune
        if "commune" not in st.session_state:
            st.session_state["commune"] = "--Select--"
        # Initialize session state for zip code
        if "zip_code" not in st.session_state:
            st.session_state["zip_code"] = ""

        with self.col3:
            st.subheader("Enter Feature Values:")

            # Inputs for commune and zip code
            st.text_input(
                "Enter a Zip Code:",
                key="zip_code",
                on_change=self.sync_fields,
                args=("zip_code",),
                help="The postal code of the property. Leave empty if entering commune name.",
            )
            st.selectbox(
                "Select a Commune:",
                options=["--Select--"] + self.communes,
                index=(
                    (["--Select--"] + self.communes).index(
                        st.session_state.get("commune", "--Select--")
                    )
                    if st.session_state.get("commune", "--Select--")
                    in ["--Select--"] + self.communes
                    else 0
                ),
                key="commune",
                on_change=self.sync_fields,
                args=("commune",),
                help="The locality where the property is situated.",
            )

            # Add input fields for other features
            input_data = {
                "living_area": st.number_input(
                    "Enter Living Area (m²):",
                    value=None,
                    step=1,
                    help="The total floor space of the property in square meters.",
                ),
                "building_condition": st.select_slider(
                    "Select the Building Condition:",
                    options=BUILDING_CONDITIONS,
                    help="Rate the overall state of the property.",
                ),
                "subtype_of_property": st.selectbox(
                    "Select a Property Subtype:",
                    options=["--Select--"] + PROPERTY_SUBTYPES,
                    help="Choose the specific type of property.",
                ),
                "equipped_kitchen": st.selectbox(
                    "Select Kitchen Equipment:",
                    options=["--Select--"] + EQUIPPED_KITCHEN,
                    help="Indicate if the kitchen is fully equipped, semi-equipped or not equipped.",
                ),
                "terrace": st.selectbox(
                    "Select if the Property has a Terrace:",
                    options=["--Select--"] + ["No", "Yes"],
                    help="Indicate, if the property has a terrace.",
                ),
                # Add entries for commune and zip code
                "commune": st.session_state["commune"],
                "zip_code": st.session_state["zip_code"],
            }
            st.write("")

            # Prediction button
            if st.button("Predict Price"):
                # Check if all values are filled in
                if any(
                    value in [None, "--Select--", 0] for value in input_data.values()
                ):
                    st.error("Please fill in all fields before submitting.")
                else:
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
                exported_data=self.data,
                input_data=input_data,
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
        # Default coordinates for Belgium
        be_lat, be_lon = 50.8503, 4.3517  # Apr. center of Belgium

        with self.col4:
            # Check, if a commune has been selected
            if selected_commune and selected_commune != "--Select--":
                # Focus on selected commune
                commune_coordinates = self.data[
                    self.data["commune"] == selected_commune
                ][["latitude", "longitude"]]
                if not commune_coordinates.empty:
                    lat = commune_coordinates.iloc[0]["latitude"]
                    lon = commune_coordinates.iloc[0]["longitude"]

                else:
                    # Notify and show default view of Belgium
                    st.warning(
                        f"Coordinates for {selected_commune} not found. Showing Belgium map."
                    )
                    lat, lon = be_lat, be_lon

            else:
                # Default view of Belgium
                lat, lon = be_lat, be_lon

            # Create a folium map centered around the commune coordinates
            m = folium.Map(location=[lat, lon], zoom_start=8)
            # Add tile layer
            folium.TileLayer(
                "CartoDB voyager",
                attr="Map tiles by CartoDB, under CC BY 3.0, Data by OpenStreetMap, under ODbL",
            ).add_to(m)

            # Add a marker if a valid commune is selected
            if (
                selected_commune
                and selected_commune != "--Select--"
                and not commune_coordinates.empty
            ):
                folium.Marker([lat, lon], popup=f"{selected_commune}").add_to(m)

            # Add toggle to switch heatmap on or off
            add_heatmap = st.toggle(
                "Show Heatmap of Regional Price Patterns (Price per sqm (€))",
                value=True,
            )
            if add_heatmap:
                # Calc price per sqm
                self.data["price_per_sqm"] = (
                    self.data["price"] / self.data["living_area"]
                )

                # Prepare heatmap data
                heatmap_data = self.data[
                    ["latitude", "longitude", "price_per_sqm"]
                ].dropna()
                heatmap_points = heatmap_data.values.tolist()

                # Add heatmap
                HeatMap(heatmap_points, radius=15).add_to(m)

                # Add a scale for the heatmap
                colormap = LinearColormap(
                    colors=["blue", "green", "yellow", "red"],
                    vmin=heatmap_data["price_per_sqm"].min(),
                    vmax=heatmap_data["price_per_sqm"].max(),
                    caption=f"Heatmap (Price per sqm (€))",
                )
                colormap.add_to(m)

            # Display map in Streamlit
            map_html = m._repr_html_()
            st.components.v1.html(map_html, height=500)


# Initialize and run the app
if __name__ == "__main__":
    main()
