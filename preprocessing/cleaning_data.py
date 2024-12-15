from typing import Any, Dict, List
from config import (
    REQUIRED_COLUMNS,
    FEATURE_ORDER,
    CONDITION_ENCODING,
    KITCHEN_ENCODING,
    SUBTYPE_MAPPING,
    SUBTYPE_ENCODING,
    SCALER_PATH,
)
import pandas as pd
import pickle
import streamlit as st


class Preprocessor:
    """Preprocesses input data for the price prediction model."""

    @classmethod
    def get_unique_values(cls, data: pd.DataFrame, column: str) -> List[str]:
        """
        Extracts a list of unique values from a specific column in a DataFrame.

        Args:
            column (str): Name of column from which to extract unique values.
            import_path (str): Path and filename of the CSV file to be imported.

        Returns:
            List[str]: List of commune names.
        """
        # Get and return list of unique values
        return data[column].unique().tolist()

    @classmethod
    def manual_mapping(
        cls, df: pd.DataFrame, column: str, mapping: Dict[str, Any]
    ) -> pd.DataFrame:
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
        with open(import_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # Reorder features
        df = df[FEATURE_ORDER]

        # Scale data
        scaled_data = scaler.transform(df.values)  # Convert DataFrame to NumPy array

        return scaled_data

    @classmethod
    def preprocess(
        cls, exported_data: pd.DataFrame, input_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Preprocesses the input data for the ML model

        Args:
            exported_data (pd.DataFrame): Data exported from training the machine learning model to include
            input_data (Dict[str, Any]): Raw input data for a house.

        Returns:
            pd.DataFrame: Preprocessed data ready for prediction.

        Raises:
            ValueError: If required data is missing.
        """
        # Validate required fields
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in input_data]
        if missing_columns:
            raise ValueError(f"Missing required fields: {', '.join(missing_columns)}")

        # Select columns from exported data to merge
        cols_to_merge = [
            "commune",
            "latitude",
            "longitude",
            "min_distance",
            "com_avg_income",
        ]
        df_to_merge = exported_data[cols_to_merge]

        # Get 'commune' from input data
        input_commune = input_data.get("commune")

        # Check if commune exists in df_to_merge
        if input_commune in df_to_merge["commune"].values:
            # Lookup the first row matching the input commune
            matched_row = df_to_merge.loc[df_to_merge["commune"] == input_commune].iloc[
                0
            ]

            # Create a new DataFrame with the combined information
            combined_data = {
                **input_data,
                **matched_row.to_dict(),
            }  # Merge dictionaries

            # Convert combined data to DataFrame
            combined_df = pd.DataFrame([combined_data])

        else:
            # If commune not in data
            raise ValueError(f"Commune '{input_commune}' not found in the data.")

        # Convert terrace to binary
        pd.set_option(
            "future.no_silent_downcasting", True
        )  # Set global downcasting option
        combined_df["terrace"] = (
            combined_df["terrace"]
            .replace({"Yes": 1, "No": 0})
            .infer_objects(copy=False)
        )  # Infer to better data type without returning a copy

        # Encode 'building_condition', 'subtype_of_property', 'equipped_kitchen' using manual label encoding as in ML
        encoded_df = cls.manual_mapping(
            combined_df, column="building_condition", mapping=CONDITION_ENCODING
        )
        encoded_df = cls.manual_mapping(
            combined_df, column="subtype_of_property", mapping=SUBTYPE_MAPPING
        )
        encoded_df = cls.manual_mapping(
            combined_df, column="subtype_of_property", mapping=SUBTYPE_ENCODING
        )
        encoded_df = cls.manual_mapping(
            combined_df, column="equipped_kitchen", mapping=KITCHEN_ENCODING
        )

        # Scale input data (selects columns to be kept and scaled)
        scaled_data = cls.scale_data(encoded_df, import_path=SCALER_PATH)

        # return preprocessed DataFrame
        return scaled_data
