DATA_PATH = "./data/exported_data.csv"

MODEL_PATH = "./model/knn_model.pkl"

SCALER_PATH = "./model/knn_scaler.pkl"

REQUIRED_COLUMNS = [
    "living_area",
    "commune",
    "building_condition",
    "subtype_of_property",
    "equipped_kitchen",
    "terrace",
]

FEATURE_ORDER = [
    "living_area",
    "com_avg_income",
    "building_condition",
    "subtype_of_property",
    "latitude",
    "longitude",
    "equipped_kitchen",
    "min_distance",
    "terrace",
]

BUILDING_CONDITIONS = ["To Restore", "To Renovate", "To Refresh", "Good", "Just Renovated", "As New"]

PROPERTY_SUBTYPES = [
    "Kot",
    "Apartment",
    "Ground Floor",
    "Flat Studio",
    "Service Flat",
    "Farmhouse",
    "Mixed Use Building",
    "Triplex",
    "Duplex",
    "House",
    "Town House",
    "Bungalow",
    "Chalet",
    "Country Cottage",
    "Apartment Block",
    "Other Property",
    "Loft",
    "Mansion",
    "Penthouse",
    "Villa",
    "Manor House",
    "Castle",
    "Exceptional Property",
]

EQUIPPED_KITCHEN = ["Not Installed", "Semi Equipped", "Installed", "Hyper Equipped"]

KITCHEN_ENCODING = {
    "Hyper Equipped": 3,
    "Installed": 2,
    "Semi Equipped": 1,
    "Not Installed": 0,
}

CONDITION_ENCODING = {
    "As New": 5,
    "Just Renovated": 4,
    "Good": 3,
    "To Refresh": 2,
    "To Renovate": 1,
    "To Restore": 0,
}

SUBTYPE_MAPPING = {
    "Villa": "luxury",
    "Exceptional Property": "luxury",
    "Manor House": "luxury",
    "Castle": "luxury",
    "Penthouse": "luxury",
    "Mansion": "luxury",
    "Loft": "luxury",
    "Other Property": "other",
    "Apartment Block": "other",
    "Duplex": "house",
    "Country Cottage": "house",
    "Chalet": "house",
    "Bungalow": "house",
    "Town House": "house",
    "House": "house",
    "Triplex": "house",
    "Mixed Use Building": "mixed use building",
    "Farmhouse": "mixed use building",
    "Kot": "apartment",
    "Flat Studio": "apartment",
    "Service Flat": "apartment",
    "Ground Floor": "apartment",
    "Apartment": "apartment",
}

SUBTYPE_ENCODING = {
    "luxury": 4,
    "other": 3,
    "house": 2,
    "mixed use building": 1,
    "apartment": 0,
}
