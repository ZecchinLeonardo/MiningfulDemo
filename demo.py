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
import plotly.express as px


# AWS Credentials
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
# 4) Plot variable distributions (original)
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
    sns.heatmap(
        corr,
        annot=False,     # <- Disable numeric annotations
        cmap='coolwarm',
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
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
# 9) Additional: Compare distribution window vs. overall
# -------------------------------------------------------
def plot_distribution_comparison(df_all: pd.DataFrame, df_window: pd.DataFrame):
    """
    Plots the density comparison of the current window data vs. the entire dataset
    for a few selected columns.
    """
    columns_to_plot = ['raw_in_left', 'raw_in_right', 'raw_out_left', 'raw_out_right']
    
    fig, axs = plt.subplots(1, len(columns_to_plot), figsize=(20, 5))
    
    for i, col in enumerate(columns_to_plot):
        ax = axs[i]
        if col in df_all.select_dtypes(include=[np.number]).columns:
            df_all[col].dropna().plot(kind='density', ax=ax, label='Overall', legend=False)
        if col in df_window.select_dtypes(include=[np.number]).columns:
            df_window[col].dropna().plot(kind='density', ax=ax, label='Window', legend=False)
        
        ax.set_title(f'Density: {col}')
        ax.legend()
    st.pyplot(fig)

# -------------------------------------------------------
# 10) Additional: Plot actual vs. predicted time series
# -------------------------------------------------------
def plot_timeseries_with_prediction(df: pd.DataFrame, time_col='timestamp',
                                    actual_col='moisture_in_z0',
                                    predicted_col='predicted_moisture'):
    """
    Plots a time series chart comparing actual vs. predicted moisture.
    Only plots rows where predicted_moisture is available.
    """
    if predicted_col not in df.columns or df[predicted_col].dropna().empty:
        st.write("No predictions to plot yet.")
        return
    
    # Filter to only rows that have non-NaN predictions
    df_filtered = df.dropna(subset=[predicted_col])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_filtered[time_col], df_filtered[actual_col], label='Actual', color='blue')
    ax.plot(df_filtered[time_col], df_filtered[predicted_col], label='Predicted', color='red')
    ax.set_xlabel("Time")
    ax.set_ylabel("Moisture")
    ax.set_title("Actual vs. Predicted Moisture Over Time")
    ax.legend()
    st.pyplot(fig)

def plot_timeseries_with_prediction_interactive(
    df: pd.DataFrame, 
    time_col='timestamp',
    actual_col='moisture_in_z0',
    predicted_col='predicted_moisture'
):
    """
    Plots an interactive time series chart comparing actual vs. predicted moisture
    using Plotly. Only plots rows where predicted_moisture is available.
    """
    if predicted_col not in df.columns or df[predicted_col].dropna().empty:
        st.write("No predictions to plot yet.")
        return

    # Filter to only rows that have non-NaN predictions
    df_filtered = df.dropna(subset=[predicted_col])

    # Construct a Plotly figure with two lines
    fig = px.line(
        df_filtered,
        x=time_col,
        y=[actual_col, predicted_col],
        labels={"value": "Moisture", "variable": "Series", time_col: "Time"},
        title="Actual vs. Predicted Moisture Over Time"
    )

    # Show it in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# 8) Streamlit App
# -------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
st.image("res/Miningful_NoBG_WhiteText.png", width=150)
st.title("Miningful Predictive Maintenance Demo")

# 8.1) Sidebar config
st.sidebar.header("⚙️ Configuration")
data_window_size = st.sidebar.slider("Data Window Size (days):", 3, 7, 3, key="data_window_slider")
polling_interval = timedelta(seconds=10)

# 8.2) Load data & train model (only once)
if "stream_data" not in st.session_state:
    # data_path = "datirs_SK.csv"
    # st.session_state.stream_data = load_data(data_path)
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

if "initial_window_set" not in st.session_state:
    st.session_state.initial_window_set = False

# We'll store the "show_graphs" state for the data exploration tab if desired
if "show_graphs" not in st.session_state:
    st.session_state.show_graphs = False

# -------------------------------------------------------
# 8.4) On first load (or whenever the user moves the slider),
#      define an "initial" window
# -------------------------------------------------------
def set_initial_window():
    """Sets the initial window so that we start with a full data_window_size of data."""
    dataset_start = df_all['timestamp'].min()
    initial_end_time = dataset_start + timedelta(days=data_window_size)
    idx = find_index_for_time(df_all, initial_end_time)
    st.session_state.current_index = idx
    st.session_state.initial_window_set = True

if not st.session_state.initial_window_set:
    set_initial_window()

# 8.5) Show model info
if st.session_state.model is not None:
    st.success("Model training complete!")
    st.metric("Mean Squared Error", round(st.session_state.model_mse, 2))
else:
    st.warning("Not enough data to train the model.")

# -------------------------------------------------------
# Create Tabs
# -------------------------------------------------------
tab1, tab2 = st.tabs(["Predictions", "Data Exploration"])

# =======================================================
# TAB 1: DATA EXPLORATION
# =======================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distributions_custom(df: pd.DataFrame, columns_to_plot: list):
    """Plot distributions for the user-selected numeric columns."""
    if not columns_to_plot:
        st.write("No columns selected.")
        return
    
    # Set up a grid of subplots depending on how many columns we have
    num_cols = len(columns_to_plot)
    fig, axs = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))

    # If there's only one column selected, axs might not be an array. Make it a list.
    if num_cols == 1:
        axs = [axs]

    for ax, col in zip(axs, columns_to_plot):
        if df[col].dropna().empty:
            ax.text(0.5, 0.5, f"No data for {col}", ha='center', va='center')
        else:
            df[col].dropna().plot(kind='density', ax=ax)
        ax.set_title(f"{col} Distribution")
    st.pyplot(fig)

def plot_correlation_custom(df: pd.DataFrame, columns_to_plot: list):
    """Plot correlation heatmap for only the user-selected columns."""
    if not columns_to_plot:
        st.write("No columns selected for correlation.")
        return

    # Filter to numeric columns & user selection
    sub_df = df[columns_to_plot].dropna(axis=0, how='any')
    if sub_df.shape[1] < 2:
        st.write("Not enough columns selected for correlation heatmap.")
        return

    corr = sub_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Selected Columns)")
    st.pyplot(fig)

# -------------------------------------------------------
# Inside your "Data Exploration" tab, you might do:
# -------------------------------------------------------
def data_exploration_tab(df_all: pd.DataFrame):
    st.subheader("Data Exploration")

    # 1) Identify numeric columns
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()

    # 2) Let the user pick columns
    selected_columns = st.multiselect(
        "Select columns to visualize",
        numeric_cols,
        default=numeric_cols[:3]  # e.g. pre-select first 3 for convenience
    )

    # 3) Distributions
    st.write("###Distributions")
    plot_distributions_custom(df_all, selected_columns)

    # 4) Correlation
    st.write("###Correlation")
    plot_correlation_custom(df_all, selected_columns)
    
with tab2:
    data_exploration_tab(df_all)


# =======================================================
# TAB 2: PREDICTIONS
# =======================================================
with tab1:
    st.subheader("Predictions and Real-Time Stream")

    # Start/Stop streaming
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.streaming:
            if st.button("▶️ Start Streaming"):
                st.session_state.streaming = True
                st.rerun()
        else:
            if st.button("🛑 Stop Streaming"):
                st.session_state.streaming = False
                st.rerun()

    # Streaming logic
    if st.session_state.streaming:
        st.write("**Streaming in progress...**")

        # If we're within bounds
        if 0 <= st.session_state.current_index < len(df_all):
            data_window = get_window_data(df_all, st.session_state.current_index, data_window_size)
            data_with_preds = predict(st.session_state.model, data_window)

            # Optionally show the last few predictions in a small table
            st.write("Last few predictions in the current window:")
            st.dataframe(data_with_preds.tail(5))  # smaller table

            # Plot time series (actual vs. predicted) for the current window
            plot_timeseries_with_prediction_interactive(data_with_preds)

            # Simulate real-time
            time.sleep(polling_interval.total_seconds())
            st.session_state.current_index += 1
            st.rerun()

        else:
            st.warning("Reached the end of the dataset.")
            st.session_state.streaming = False

    else:
        st.write("**Streaming is paused.**")
        # Show predictions for the current window if available
        end_idx = min(st.session_state.current_index, len(df_all) - 1)
        data_window = get_window_data(df_all, end_idx, data_window_size)
        data_with_preds = predict(st.session_state.model, data_window)

        st.write("Last few predictions in the current window:")
        if not data_with_preds.empty:
            st.dataframe(data_with_preds.tail(5))  # smaller table
        else:
            st.write("No data available in the current window.")

        # Plot time series (actual vs predicted) for the current window
        plot_timeseries_with_prediction_interactive(data_with_preds)

st.write("""
    <script>
        // Scroll to top of page on load
        window.scrollTo(0, 0);
    </script>
""", unsafe_allow_html=True)
