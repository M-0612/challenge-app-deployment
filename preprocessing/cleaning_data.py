# Preprocesses received data

"""
This will implement functions to clean and preprocess the incoming data (e.g., handling missing values, encoding categorical features, etc.).

Define preprocess() function that:
- Takes input data for a house (e.g. location, living area, type of property, etc.)
- Handles missing or incorrect data and returns an error to the user if required information is missing.
- Returns cleaned data that is ready for the model
"""

from typing import Any, Dict, List
from config import CONDITION_ENCODING, KITCHEN_ENCODING, SUBTYPE_MAPPING, SUBTYPE_ENCODING, SCALER_PATH
import pandas as pd
import numpy as np
import pickle

class Preprocessor:
    """Preprocesses input data for the price prediction model."""

    # Initialize required input columns
    REQUIRED_COLUMNS = ['living_area', 'commune', 'building_condition', 'subtype_of_property', 'equipped_kitchen', 'terrace']

    @classmethod
    def get_data(cls, import_path: str) -> pd.DataFrame:
        """
        Imports commune data (commune names, latitude, longitude, distance to nearest large city).

        Args:
            import_path (str): Path and filename of the CSV file to be imported.

        Returns:
            pd.DataFrame: The imported data as DataFrame.
        """
        # Import and return commune data as DataFrame
        return pd.read_csv(import_path)
    
    @classmethod
    def get_unique_values(cls, column: str, import_path: str) -> List[str]:
        """
        Extracts a list of unique values from a specific column in a DataFrame.

        Args:
            column (str): Name of column from which to extract unique values.
            import_path (str): Path and filename of the CSV file to be imported.
        
        Returns:
            List[str]: List of commune names.
        """
        df = cls.get_data(import_path)

        # Get and return list of unique values
        return df[column].unique().tolist()
    
    @classmethod
    def manual_mapping(cls, df: pd.DataFrame, column: str, mapping: Dict[str, Any]) -> pd.DataFrame:
        """
        Replaces categorical values in a given column based on a given mapping.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column to apply the mapping to.
            mapping (Dict[Str, Any]): Mapping values to apply to the column.
        
        Returns:
            pd.DataFrame: DataFrame with replaced categorical values.
        """
        df[column] = df[column].map(mapping)
        return df
    
    @classmethod
    def scale_data(cls, df: pd.DataFrame, import_path: str) -> pd.DataFrame:
        """
        Standardizes the input data using the 'trained' scaler.

        Args:
            df (pd.DataFrame): Input DataFrame
            import_path (str): Path and filename to load the scaler from.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        # Load saved scaler
        with open(import_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        
        # Scale data
        scaled_data = scaler.transform(df)

        return scaled_data

    @classmethod
    def preprocess(cls, data: Dict[str, Any], import_path: str) -> pd.DataFrame:
        """
        Preprocesses the input data for the ML model

        Args:
            data (Dict[str, Any]): Raw input data for a house.
        
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction.
        
        Raises:
            ValueError: If required data is missing.
        """
        # Validate required fields
        missing_columns = [col for col in cls.REQUIRED_COLUMNS if col not in data]
        if missing_columns:
            raise ValueError(f"Missing required fields: {', '.join(missing_columns)}")

        # Convert input data (dict) to DataFrame
        df = pd.DataFrame([data])

        # Fill NaN values # TODO: use or remove
        #df.fillna({  }, inplace=True)

        # Get commune data
        commune_data_df = cls.get_data(import_path)

        # Merge the DataFrames based on the 'commune' column
        merged_df = pd.merge(df, commune_data_df, on='commune', how='left')

        # Drop 'commune' column
        merged_df = merged_df.drop(columns=['commune'])

        # Convert terrace to binary
        merged_df['terrace'] = df['terrace'].replace({'Yes': 1, 'No': 0})

        # Encode 'building_condition', 'subtype_of_property', 'equipped_kitchen' using manual label encoding as in ML
        encoded_df = cls.manual_mapping(merged_df, column='building_condition', mapping=CONDITION_ENCODING)
        encoded_df = cls.manual_mapping(encoded_df, column='subtype_of_property', mapping=SUBTYPE_MAPPING)
        encoded_df = cls.manual_mapping(encoded_df, column='subtype_of_property', mapping=SUBTYPE_ENCODING)
        encoded_df = cls.manual_mapping(encoded_df, column='equipped_kitchen', mapping=KITCHEN_ENCODING)

        # Scale input data
        scaled_df = cls.scale_data(encoded_df, import_path=SCALER_PATH) # TODO: fit_transform or transform?

        # return list of communes and preprocessed DataFrame
        return scaled_df