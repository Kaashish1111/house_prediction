import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Load and prepare data (only once)
# -------------------------------
@st.cache_resource
def load_data_and_train():
    df = pd.read_csv("Bengaluru_House_Data.csv")

    # Preprocessing
    df = df.drop(["area_type", "society", "availability"], axis=1)
    df = df.dropna()
    df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))

    def convert_sqft(x):
        try:
            if "-" in str(x):
                a, b = x.split("-")
                return (float(a) + float(b)) / 2
            return float(x)
        except:
            return None

    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    df = df.dropna(subset=["total_sqft"])

    np.random.seed(42)
    df["property_age"] = np.random.randint(0, 31, size=len(df))
    df["amenities_score"] = np.random.randint(0, 11, size=len(df))

    df = df[["location", "total_sqft", "bhk", "property_age", "amenities_score", "bath", "price"]]

    le = LabelEncoder()
    df["location"] = le.fit_transform(df["location"])

    # Features & Target
    X = df.drop("price", axis=1)
    y = df["price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest (best performing model from your notebook)
    model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X_scaled, y)

    return model, scaler, le

model, scaler, le = load_data_and_train()

# -------------------------------
# Streamlit UI
# -------------------------------


st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #FFFFFF;
        }
        [data-testid="stAppViewContainer"] {
    background-image: url("https://media.istockphoto.com/id/824894330/photo/vidhana_soudha.webp?a=1&b=1&s=612x612&w=0&k=20&c=hNj5NQrQtrpjKGbetMzom2fKqZppHZRNppO2UV1af20=");
    background-size: cover;
    background-position: center;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
        .stButton>button {
    background-color: #E63946; /* red */
    color: white; /* text color */
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #B71C1C; /* darker red on hover */
    color: white;
}


        div.stAlert.success {
            background-color: ##FF1F15 /* light coral */
            color: #34495e /* keep text readable */
            border-radius: 12px;
            padding: 16px;
        }

    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>House Price Prediction</h1>", unsafe_allow_html=True)

# Input fields
location = st.text_input("Location (e.g., Whitefield, Indiranagar)")
total_sqft = st.number_input("Total Square Feet", min_value=200, step=50)
bhk = st.number_input("Number of BHK", min_value=1, step=1)
property_age = st.slider("Property Age (years)", 0, 30, 5)
amenities_score = st.slider("Amenities Score (0 = poor, 10 = luxury)", 0, 10, 5)
bath = st.number_input("Number of Bathrooms", min_value=1, step=1)

if st.button("Predict Price"):
    # Encode location
    loc_code = le.transform([location])[0] if location in le.classes_ else 0

    # Prepare features
    features = np.array([[loc_code, total_sqft, bhk, property_age, amenities_score, bath]])
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    st.success(f"💰 Estimated Price: {round(prediction, 2)} Lakhs")

    st.caption("Kashish generated this predetion yoooo!!!!!!")
