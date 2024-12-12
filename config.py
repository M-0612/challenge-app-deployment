MODEL_PATH = "./model/knn_model.pkl"

COMMUNE_DATA_PATH = "./data/commune_data.csv"

SCALER_PATH = "./model/knn_scaler.pkl"

BUILDING_CONDITIONS = ["to restore", "to renovate", "good", "just renovated", "as new"]

PROPERTY_SUBTYPES = ["kot", "apartment", "ground floor", "flat studio", "service flat", "farmhouse", "mixed use building", "triplex", "duplex", "house", "town house", "bungalow", "chalet", "country cottage", "apartment block", "other property", "loft", "mansion", "penthouse", "villa", "manor house", "castle", "exceptional property"]

EQUIPPED_KITCHEN = ['not installed', 'semi equipped', 'installed', 'hyper equipped']

KITCHEN_ENCODING = {
    "hyper equipped": 3,
    "installed": 2,
    "semi equipped": 1,
    "not installed": 0,
}

CONDITION_ENCODING = {
    "as new": 5,
    "just renovated": 4,
    "good": 3,
    "to be done up": 2,
    "to renovate": 1,
    "to restore": 0,
}

SUBTYPE_MAPPING = {
    "villa": "luxury",
    "exceptional property": "luxury",
    "manor house": "luxury",
    "castle": "luxury",
    "penthouse": "luxury",
    "mansion": "luxury",
    "loft": "luxury",
    "other property": "other",
    "apartment block": "other",
    "duplex": "house",
    "country cottage": "house",
    "chalet": "house",
    "bungalow": "house",
    "town house": "house",
    "house": "house",
    "triplex": "house",
    "mixed use building": "mixed use building",
    "farmhouse": "mixed use building",
    "kot": "apartment",
    "flat studio": "apartment",
    "service flat": "apartment",
    "ground floor": "apartment",
    "apartment": "apartment"
}

SUBTYPE_ENCODING = {
    "luxury": 4,
    "other": 3,
    "house": 2,
    "mixed use building": 1,
    "apartment": 0,
}