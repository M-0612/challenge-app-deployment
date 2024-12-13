"""will contain all the code used to predict a new house's price.
Your file should contain a function predict() that will take your preprocessed data as an input and return a price as output."""

import pickle
import pandas as pd
from typing import Any


class Predictor:
    """
    Class for loading a trained machine learning model and predicting real-estate prices.
    """

    def __init__(
        self, model_path: str
    ):  # Encapsulates the model-loading logic into the initialization phase
        """
        Initializes the predictor by loading the trained model.

        Args:
            model_path (str): Path to the saved model file.
        """
        self.model = self.load_model(model_path)

    @staticmethod  # In case the model is required outside the class
    def load_model(model_path: str) -> Any:
        """
        Loads the trained machine learning model from the specified path.

        Args:
            model_path (str): Path to the saved model file.

        Returns:
            Any: The trained model.
        """
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        return model

    def predict(self, scaled_data: Any) -> pd.Series:
        """
        Predicts real-estate prices based on the scaled input data.

        Args:
            scaled_data (Any): Scaled features as a NumPy array.

        Returns:
            pd.Series: Predicted real-estate prices.
        """
        # Make predictions
        predictions = self.model.predict(scaled_data)

        # Return prediction as a pandas Series for easier manipulation
        return pd.Series(predictions, name="Predicted_Price")
