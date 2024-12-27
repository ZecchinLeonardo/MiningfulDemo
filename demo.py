import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
import boto3
import pandas as pd
from io import StringIO


aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)


# -------------------------------------------------------
# 1) Load the entire CSV once
# -------------------------------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path, sep=';', parse_dates=['to_timestamp'])
    df.rename(columns={'to_timestamp': 'timestamp'}, inplace=True)
    df.sort_values('timestamp', inplace=True, ignore_index=True)
    return df

def load_data_remote():
    obj = s3.get_object(Bucket='miningfuldemo', Key='datirs_SK.csv')
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data), sep=';', parse_dates=['to_timestamp'])
    df.rename(columns={'to_timestamp': 'timestamp'}, inplace=True)
    df.sort_values('timestamp', inplace=True, ignore_index=True)
    return df

# -------------------------------------------------------
# 2) Train a simple model on the entire dataset
# -------------------------------------------------------
def train_model(df: pd.DataFrame):
    feature_columns = [
        'raw_in_left', 'raw_in_right', 'raw_out_left', 'raw_out_right',
        'paperwidth_in', 'paperwidth_out', 'temp_in_z0', 'temp_out_z0',
        'temp_in_z1', 'temp_out_z1', 'temp_in_z2', 'temp_out_z2',
        'temp_in_z3', 'temp_out_z3', 'temp_in_z4', 'temp_out_z4',
        'temp_in_z5', 'temp_out_z5', 'temp_in_z6', 'temp_out_z6',
        'temp_in_z7', 'temp_out_z7', 'temp_in_z8', 'temp_out_z8',
        'temp_in_z9', 'temp_out_z9'
    ]
    target_column = 'moisture_in_z0'

    df_model = df.dropna(subset=feature_columns + [target_column])
    if df_model.empty:
        return None, None

    X = df_model[feature_columns]
    y = df_model[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse

# -------------------------------------------------------
# 3) Predict on a given window
# -------------------------------------------------------
def predict(model, data_window: pd.DataFrame) -> pd.DataFrame:
    if model is None or data_window.empty:
        return data_window

    feature_columns = [
        'raw_in_left', 'raw_in_right', 'raw_out_left', 'raw_out_right',
        'paperwidth_in', 'paperwidth_out', 'temp_in_z0', 'temp_out_z0',
        'temp_in_z1', 'temp_out_z1', 'temp_in_z2', 'temp_out_z2',
        'temp_in_z3', 'temp_out_z3', 'temp_in_z4', 'temp_out_z4',
        'temp_in_z5', 'temp_out_z5', 'temp_in_z6', 'temp_out_z6',
        'temp_in_z7', 'temp_out_z7', 'temp_in_z8', 'temp_out_z8',
        'temp_in_z9', 'temp_out_z9'
    ]
    X = data_window[feature_columns].dropna()
    if X.empty:
        return data_window

    preds = model.predict(X)
    data_window = data_window.copy()
    data_window.loc[X.index, 'predicted_moisture'] = preds
    return data_window

# -------------------------------------------------------
# 4) Plot variable distributions
# -------------------------------------------------------
def plot_distributions(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    columns_to_plot = ['raw_in_left', 'raw_in_right', 'raw_out_left', 'raw_out_right']

    for i, col in enumerate(columns_to_plot):
        if col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dropna().empty:
                ax[i].text(0.5, 0.5, f'No data for {col}', ha='center', va='center')
            else:
                df[col].plot(kind='density', ax=ax[i], legend=False)
            ax[i].set_title(f'{col} Distribution')
    st.pyplot(fig)

# -------------------------------------------------------
# 5) Plot correlation heatmap
# -------------------------------------------------------
def plot_correlation(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if numeric_df.shape[1] < 2:
        st.write("Not enough numeric columns for a correlation heatmap.")
        return
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

# -------------------------------------------------------
# 6) Helper: find the row index that matches or exceeds a given timestamp
# -------------------------------------------------------
def find_index_for_time(df: pd.DataFrame, t: pd.Timestamp) -> int:
    """Return the first row index where df['timestamp'] >= t, or the last row if none."""
    matching = df.index[df['timestamp'] >= t]
    return matching[0] if len(matching) > 0 else len(df) - 1

# -------------------------------------------------------
# 7) Helper: get window data from index/time
# -------------------------------------------------------
def get_window_data(df, end_index: int, days: int):
    """Return rows in [end_time - days, end_time], where end_time is at `end_index` row."""
    if end_index < 0 or end_index >= len(df):
        return pd.DataFrame()
    end_time = df.loc[end_index, 'timestamp']
    start_time = end_time - timedelta(days=days)
    return df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

# -------------------------------------------------------
# 8) Streamlit App
# -------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
st.title("🚀 Miningful Predictive Maintenance Demo")

# 8.1) Sidebar config
st.sidebar.header("⚙️ Configuration")
data_window_size = st.sidebar.slider("Data Window Size (days):", 3, 7, 3, key="data_window_slider")
polling_interval = timedelta(seconds=10)

# 8.2) Load data & train model (only once)
if "stream_data" not in st.session_state:
    data_path = "datirs_SK.csv"
    #st.session_state.stream_data = load_data(data_path)
    st.session_state.stream_data = load_data_remote()

df_all = st.session_state.stream_data

if "model" not in st.session_state:
    with st.spinner("Training model..."):
        model, mse = train_model(df_all)
        st.session_state.model = model
        st.session_state.model_mse = mse

# 8.3) Initialize session state
if "streaming" not in st.session_state:
    st.session_state.streaming = False

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# We track if we've set an "initial" index after changing the window size or first load
# so we can show a full initial window right away.
if "initial_window_set" not in st.session_state:
    st.session_state.initial_window_set = False

if "show_graphs" not in st.session_state:
    st.session_state.show_graphs = False

# -------------------------------------------------------
# 8.4) On first load (or whenever the user moves the slider),
#      define an "initial" window: [start_of_dataset, start_of_dataset + data_window_size]
#      Then set current_index to that "end_time" row.
# -------------------------------------------------------
def set_initial_window():
    """Sets the initial window so that we start with a full data_window_size of data."""
    dataset_start = df_all['timestamp'].min()
    # End time is dataset_start + data_window_size
    initial_end_time = dataset_start + timedelta(days=data_window_size)
    # Find the row index matching that end time
    idx = find_index_for_time(df_all, initial_end_time)
    st.session_state.current_index = idx
    st.session_state.initial_window_set = True

# Run the above if not set or if the user just changed the slider
# A naive approach: we always set it if it hasn't been set yet.
if not st.session_state.initial_window_set:
    set_initial_window()

# -------------------------------------------------------
# 8.5) Show model info
# -------------------------------------------------------
if st.session_state.model is not None:
    st.success("Model training complete!")
    st.metric("Mean Squared Error", round(st.session_state.model_mse, 2))
else:
    st.warning("Not enough data to train the model.")

# -------------------------------------------------------
# 8.6) Start/Stop streaming with immediate rerun to avoid double-click
# -------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    if not st.session_state.streaming:
        if st.button("▶️ Start Streaming"):
            st.session_state.streaming = True
            st.rerun()
    else:
        if st.button("🛑 Stop Streaming"):
            st.session_state.streaming = False
            st.session_state.show_graphs = False
            st.rerun()

# -------------------------------------------------------
# 8.7) Handle streaming
# -------------------------------------------------------
if st.session_state.streaming:
    st.subheader("📡 Real-time Data Stream (Streaming in progress)")

    if 0 <= st.session_state.current_index < len(df_all):
        data_window = get_window_data(df_all, st.session_state.current_index, data_window_size)
        data_with_preds = predict(st.session_state.model, data_window)
        st.dataframe(data_with_preds.tail(20))

        time.sleep(polling_interval.total_seconds())  # simulate real time
        st.session_state.current_index += 1
        st.rerun()

    else:
        st.warning("Reached the end of the dataset.")
        st.session_state.streaming = False

# -------------------------------------------------------
# 8.8) Paused (not streaming)
# -------------------------------------------------------
else:
    st.subheader("📡 Real-time Data Stream (Paused)")

    if len(df_all) > 0:
        end_idx = min(st.session_state.current_index, len(df_all) - 1)
        data_window = get_window_data(df_all, end_idx, data_window_size)
        data_with_preds = predict(st.session_state.model, data_window)

        st.write(f"**Current Window ({len(data_with_preds)} rows)**")
        st.dataframe(data_with_preds)
    else:
        data_window = pd.DataFrame()
        st.write("No data loaded.")

    if st.button("Generate Graphs"):
        st.session_state.show_graphs = True
        st.rerun()

    if st.session_state.show_graphs:
        st.subheader("🔍 Data Insights")
        st.write("### Sensor Data Distributions")
        if not data_window.empty:
            plot_distributions(data_window)
        else:
            st.write("No data available.")

        st.write("### Data Correlation")
        if not data_window.empty:
            plot_correlation(data_window)
        else:
            st.write("No data available.")
