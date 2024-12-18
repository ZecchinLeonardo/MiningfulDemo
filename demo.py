import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simulate data generation
def generate_fake_data(timestamp: datetime) -> list:
    sensor_1 = np.random.normal(50, 5)
    sensor_2 = np.random.normal(100, 10)
    sensor_3 = np.random.normal(75, 7)
    machine_status = np.random.choice([0, 1], p=[0.95, 0.05])  # 0 = normal, 1 = failure
    return [timestamp, sensor_1, sensor_2, sensor_3, machine_status]

# Initialize streaming data
def initialize_stream_data(data_window_size: int, update_interval: int) -> pd.DataFrame:
    total_data_points = (data_window_size * 60) // update_interval
    base_time = datetime.now() - timedelta(seconds=total_data_points * update_interval)
    data = [
        generate_fake_data(base_time + timedelta(seconds=i * update_interval))
        for i in range(total_data_points)
    ]
    return pd.DataFrame(data, columns=['timestamp', 'sensor_1', 'sensor_2', 'sensor_3', 'machine_status'])

# Train a simple model on historical data
def train_model(df: pd.DataFrame):
    X = df[['sensor_1', 'sensor_2', 'sensor_3']]
    y = df['machine_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, classification_report(y_test, model.predict(X_test))

# Predict on new data
def predict(model, data_window: pd.DataFrame) -> pd.DataFrame:
    if data_window.empty:
        return data_window
    X = data_window[['sensor_1', 'sensor_2', 'sensor_3']]
    predictions = model.predict(X)
    data_window = data_window.copy()
    data_window['prediction'] = predictions
    return data_window

# Streamlit app
st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
st.title("🚀 Miningful Predictive Maintenance Demo")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
update_interval = st.sidebar.slider("Update Interval (seconds):", 1, 60, 3)
data_window_size = st.sidebar.slider("Data Window Size (minutes):", 1, 60, 15)

# Initialize streaming data and streaming state
if "stream_data" not in st.session_state:
    st.session_state.stream_data = initialize_stream_data(data_window_size, update_interval)

if "streaming" not in st.session_state:
    st.session_state.streaming = False

# Train the model
st.subheader("📊 Model Training on Historical Data")
model, report = train_model(st.session_state.stream_data)
st.success("Model training complete!")
st.text_area("Classification Report:", report, height=200)

# Streaming controls
st.subheader("📡 Real-time Data Stream")
col1, col2 = st.columns(2)
with col1:
    if st.session_state.streaming:
        if st.button("🛑 Stop Streaming"):
            st.session_state.streaming = False
            st.rerun()
    else:
        if st.button("▶️ Start Streaming"):
            st.session_state.streaming = True
            st.session_state.stream_start_time = time.time()  # Record start time
            st.rerun()

with col2:
    st.metric("Update Interval", f"{update_interval} seconds")

# Real-time updates
if st.session_state.streaming:
    # Simulate new data arrival
    new_timestamp = st.session_state.stream_data.iloc[-1]['timestamp'] + timedelta(seconds=update_interval)
    new_data = generate_fake_data(new_timestamp)
    new_row = pd.DataFrame([new_data], columns=['timestamp', 'sensor_1', 'sensor_2', 'sensor_3', 'machine_status'])
    st.session_state.stream_data = pd.concat([st.session_state.stream_data, new_row], ignore_index=True)

    # Update data window
    current_time = datetime.now()
    start_time = current_time - timedelta(minutes=data_window_size)
    data_window = st.session_state.stream_data[st.session_state.stream_data['timestamp'] >= start_time]

    # Make predictions
    data_with_predictions = predict(model, data_window)

    # Display streaming data and predictions
    st.dataframe(
        data_with_predictions.tail(20).style.map(
            lambda x: "background-color: red; color: white;" if x == 1 else "",
            subset=['prediction']
        )
    )

    # Check for limited demo time (e.g., 5 minutes)
    if "stream_start_time" in st.session_state:
        elapsed_time = time.time() - st.session_state.stream_start_time
        if elapsed_time > 300:  # Stop after 5 minutes
            st.session_state.streaming = False
            st.warning("Demo time is over. Click '▶️ Start Streaming' to restart.")

    # Wait for the next update and rerun
    time.sleep(update_interval)
    st.rerun()

else:
    st.info("Streaming paused. Click '▶️ Start Streaming' to continue.")
