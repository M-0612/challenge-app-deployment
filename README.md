# ImmoEliza Real-Estate Price Predictor 📊🏠

Welcome to the **ImmoEliza Real-Estate Price Predictor**! 
This Streamlit app allows users to interact with a trained machine learning model to predict the price of real estate properties in Belgium, based on user entries of feature values such as for living area, building condition, location, etc. 

## Description

This app is the fourth and final phase of the *ImmoEliza project*, which is part of the BeCode Data Science and AI Bootcamp. The project's purpose is to deliver a Streamlit-based application that integrates a K-Nearest Neighbors (KNN) regression model, allowing users to estimate real estate prices interactively.

**Limitations:**

* Predictions rely on the quality of input data. Missing or inaccurate values can lead to errors.
* The app currently does not support real-time updates or external datasets.
* Predictions depend heavily on the quality and coverage of the dataset. The original dataset the KNN regression model is based on may contain imbalanced representation of different properties and features. As a result, certain feature combinations may result in inaccurate predictions.
  
* Improvements could include:
  * Expanding feature choice and countries.
  * Improving predictions by addressing data imbalances.
  * Integrating real-time market trends.
  * Adding other machine learning models for comparison and better accuracy. 

---

## Installation

Follow these steps to set up the app locally:

1. **Clone the Repository and Navigate to Directory**
   ```bash
   git clone https//github.com/M-0612/challenge-app-deployment.git

   cd challenge-app-deployment
   ```
2. **Install Dependencies**
    
    Create a virtual environment and install the required libraries:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ``` 

3. **Launch the App**

    Run the app locally:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the app in your browser (http://localhost:8501).
   
2. Fill in the property details, such as living area, location, building condition, kitchen details, etc.

3. When selecting a commune, the app displays a map showing the commune's location. A heatmap provides information regarding regional price patterns that can be turned off using the toggle switch.

4. Click "Predict Price" to see the estimated value.

## File Structure

```plaintext

challenge-app-deployment
├── .streamlit/            # Streamlit-specific configurations
│   └── config.toml
├── data/                  # Real-estate dataset for heatmap
│   └── exported_data.csv
├── images/                # App visuals (e.g., logo)
│   └── logo.png
├── model/                 # Saved KNN model and scaler
│   ├── knn_model.pkl
│   └── knn_scaler.pkl
├── predict/               # Prediction logic
│   ├── __init__.py
│   └── prediction.py
├── preprocessing/         # Data cleaning and preprocessing logic
│   ├── __init__.py
│   └── cleaning_data.py
├── app.py                 # Main Streamlit app
├── config.py              # Configuration settings for paths and options
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies

```
## Explanation of Main Scripts and Files

### Main Scripts

* `app.py`

* `preprocessing/cleaning_data.py`

* `predict/prediction.py`

### Key Files

* `data/exported_data.csv`
  * **Description:** The dataset used during machine learning containing real-estate data, including commune names, latitude, longitude, and prices.
  * **Purpose:**
    * Contains essential data (e.g. commune coordinates) for geospatial functionalities.
    * Provides commune-specific data for map visualization and heatmap creation.

* `.streamlit/config.toml`
  * **Description:** Configures the Streamlit app's appearance.
  * **Purpose:**
    * Defines custom page settings, such as wide layout, background and colors.

* `images/logo.png`
  * **Description:** A visual branding element used in the app's header.
  * **Usage:**
    * The logo can be replaced by either replacing the file while keeping the filename, or by changing the filename in `config.py`.

* `model/knn_model.pkl`
  * **Description:** The saved K-Nearest Neighbors regression model trained during the machine learning phase.
  * **Purpose and Usage:**
    * Loaded by the `Predictior` class to perform price predictions.
    * Contains the learned relationships between property features and prices.

* `model/knn_scaler.pkl`
  * **Description:** A saved scaler object used during the training phase to normalize input data.
  * **Purpose and Usage:**
    * Ensures new input data matches the scale of the training data for accurate predictions.
    * Used by the `predict()` method in `prediction.py`

* `__init__.py` Files
  * **Description:** Makes the `preprocessing` and `predict` directories Python packages.
  * **Purpose:**
    * Enables importing modules within these directories.
    * Contains no additional logic for this project.

* `config.py`
  * **Description:** Stores configuration constants used throughout the app.
  * **Purpose and Usage:**
    * Defines paths to key files (`MODEL_PATH`, `DATA_PATH`)
    * Provides reusable lists for dropdowns (e.g building conditions, property subtypes)

* `requirements.txt`
  * **Description:** Lists all Python dependencies required to run the project.
  * **Purpose and Usage:**
    * Ensures consistent setup across environments.
    * Includes essential libraries like `streamlit`, `pandas`, and `folium`.

## Visuals

App preview:

*App Screenshot*

## Contributors

* **Solo Project**
  
  Built with 💻 and ☕ by Miriam Stoehr, as part of the BeCode Data Science and AI Bootcamp 2024/2025.


## Personal Situation

This app is the fourth and final phase of the **ImmoEliza project**, developed during BeCode's Data Science and AI Bootcamp. The stages of the project included:

1. **Web scraping:** Creating a dataset based on data collected from a real estate platform.
2. **Data Analysis:** Processing and analyzing market patterns and trends.
3. **Machine Learning:** Building a Machine Learning model (my group's focus was on KNN regression) to predict property prices.
4. **Deployment:** Developing this app to deploy the model interactively.