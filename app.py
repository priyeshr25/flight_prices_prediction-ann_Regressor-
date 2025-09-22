import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor

# ==========================
# Define build_ann BEFORE loading pickle
# ==========================
def build_ann(input_dim, learning_rate=0.001):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")  # Regression output
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

# ==========================
# Load pipeline & dataset
# ==========================
pipe = pickle.load(open("pipe.pkl", "rb"))
flight_data = pickle.load(open("flight_data.pkl", "rb"))

# ==========================
# Streamlit Page Config
# ==========================
st.set_page_config(page_title="✈️ Flight Price Predictor", layout="wide")

# ==========================
# App Title & Subtitle (No HTML)
# ==========================
st.title("✈️ Flight Price Predictor")
#st.caption("Predict flight prices instantly using our ann model")
st.divider()

# ==========================
# Layout with Columns
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Flight Details")
    airline = st.selectbox("✈️ Select Airline", flight_data['airline'].unique())
    flight = st.selectbox("🆔 Flight Number", flight_data['flight'].unique())
    source = st.selectbox("📍 Source City", flight_data['source_city'].unique())
    destination = st.selectbox("🏙 Destination City", flight_data['destination_city'].unique())

with col2:
    st.subheader("Travel Info")
    departure_time = st.selectbox("🕒 Departure Time", flight_data['departure_time'].unique())
    arrival_time = st.selectbox("🛬 Arrival Time", flight_data['arrival_time'].unique())
    duration = st.selectbox("⏱ Duration", flight_data['duration'].unique())
    stops = st.selectbox("🚏 Number of Stops", flight_data['stops'].unique())
    class_type = st.selectbox("💺 Class", flight_data['class'].unique())
    days_left = st.slider("📆 Days Left for Journey", 0, 60, 30)

# ==========================
# Predict Button
# ==========================
if st.button("🔍 Predict Price", use_container_width=True):
    query = pd.DataFrame({
        "airline": [airline],
        "flight": [flight],
        "source_city": [source],
        "destination_city": [destination],
        "departure_time": [departure_time],
        "arrival_time": [arrival_time],
        "duration": [duration],
        "stops": [stops],
        "class": [class_type],
        "days_left": [days_left]
    })
    prediction = pipe.predict(query)[0]
    st.success(f" Estimated Price: ₹ {prediction:,.0f}")