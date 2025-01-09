import streamlit as st
from streamlit.runtime.fragment import fragment  # For partial re-runs
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import os
import boto3
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

###############################################################################
# AWS Credentials and S3 client
###############################################################################
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

###############################################################################
# Data loading and model training
###############################################################################
def load_data_remote():
    obj = s3.get_object(Bucket='miningfuldemo', Key='datirs_SK.csv')
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data), sep=';', parse_dates=['to_timestamp'])
    df.rename(columns={'to_timestamp': 'timestamp'}, inplace=True)
    df.sort_values('timestamp', inplace=True, ignore_index=True)
    return df

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

###############################################################################
# Prediction & Plotting Utilities
###############################################################################
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

def find_index_for_time(df: pd.DataFrame, t: pd.Timestamp) -> int:
    matching = df.index[df['timestamp'] >= t]
    return matching[0] if len(matching) > 0 else len(df) - 1

def get_window_data(df, end_index: int, days: int):
    if end_index < 0 or end_index >= len(df):
        return pd.DataFrame()
    end_time = df.loc[end_index, 'timestamp']
    start_time = end_time - timedelta(days=days)
    return df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

def plot_timeseries_with_prediction_interactive(
    df: pd.DataFrame, 
    time_col='timestamp',
    actual_col='moisture_in_z0',
    predicted_col='predicted_moisture'
):
    if predicted_col not in df.columns or df[predicted_col].dropna().empty:
        st.write("No predictions to plot yet.")
        return

    df_filtered = df.dropna(subset=[predicted_col])
    fig = px.line(
        df_filtered,
        x=time_col,
        y=[actual_col, predicted_col],
        labels={"value": "Moisture", "variable": "Series", time_col: "Time"},
        title="Actual vs. Predicted Moisture Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# Data Exploration / Visualization
###############################################################################
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

def plot_distributions_custom(df: pd.DataFrame, columns_to_plot: list):
    if not columns_to_plot:
        st.write("No columns selected.")
        return
    
    num_cols = len(columns_to_plot)
    fig, axs = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
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
    if not columns_to_plot:
        st.write("No columns selected for correlation.")
        return

    sub_df = df[columns_to_plot].dropna(axis=0, how='any')
    if sub_df.shape[1] < 2:
        st.write("Not enough columns selected for correlation heatmap.")
        return

    corr = sub_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Selected Columns)")
    st.pyplot(fig)

def plot_distribution_comparison(df_all: pd.DataFrame, df_window: pd.DataFrame):
    """
    Plots density comparison of the current window vs. the entire dataset
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

###############################################################################
# Enhanced Anomaly Detection - More User Friendly
###############################################################################

def detect_anomalies_zscore(df: pd.DataFrame, columns: list, threshold: float = 3.0):
    """
    Detect anomalies in specified columns using a simple Z-score method.
    Adds per-column anomaly flags (col_anomaly) and an 'is_anomaly' column.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std == 0:
                df[f"{col}_anomaly"] = False
            else:
                zscores = (df[col] - col_mean) / col_std
                df[f"{col}_anomaly"] = np.abs(zscores) > threshold
    
    # For a simple aggregated flag
    anomaly_cols = [c for c in df.columns if c.endswith("_anomaly")]
    if anomaly_cols:
        df['is_anomaly'] = df[anomaly_cols].any(axis=1)
    else:
        df['is_anomaly'] = False
    return df

def plot_anomalies_time_series(df: pd.DataFrame, x_col: str, y_col: str, anomaly_col='is_anomaly'):
    """
    Create a scatter chart showing normal vs. anomalous points for a single column.
    """
    if x_col not in df.columns or y_col not in df.columns:
        st.write(f"Cannot plot anomalies for {y_col}. Missing columns.")
        return

    # We'll create a color label for anomalies
    df_plot = df.copy()
    df_plot['Anomaly'] = df_plot[anomaly_col].replace({True: "Anomaly", False: "Normal"})
    
    fig = px.scatter(
        df_plot, 
        x=x_col, 
        y=y_col, 
        color='Anomaly',
        title=f"Anomaly Visualization: {y_col} vs. {x_col}"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# Feature Importance / Key Variable Contribution
###############################################################################
def plot_feature_importances(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        st.write("Model does not have feature importances.")
        return

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sorted_features, sorted_importances, color='skyblue')
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (RandomForest)")
    st.pyplot(fig)

###############################################################################
# Fragment 1: Data Exploration (Updated Anomaly Section)
###############################################################################
@fragment
def data_exploration_tab(df_all: pd.DataFrame, model, feature_columns):
    st.subheader("Data Exploration")

    # We'll consider the current window as well
    end_idx = st.session_state.current_index
    data_window = get_window_data(df_all, end_idx, st.session_state.get("demo_window_size", 3))

    # Let user select from a variety of analysis/plots:
    analysis_options = [
        "Distributions (Full Dataset)",
        "Correlation Heatmap (Full Dataset)",
        "Window Distribution Comparison",
        "Anomaly Detection (Window)",
        "Feature Importances"
    ]
    selected_analyses = st.multiselect(
        "Select Analyses to Display",
        analysis_options,
        default=["Window Distribution Comparison", "Anomaly Detection (Window)", "Feature Importances"]
    )

    # 1) Distributions (Full Dataset)
    if "Distributions (Full Dataset)" in selected_analyses:
        numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        chosen_dist_cols = st.multiselect(
            "Choose columns for distribution plots (Full Dataset)",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        st.write("### Distributions (Full Dataset)")
        plot_distributions_custom(df_all, chosen_dist_cols)

    # 2) Correlation Heatmap (Full Dataset)
    if "Correlation Heatmap (Full Dataset)" in selected_analyses:
        numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        chosen_corr_cols = st.multiselect(
            "Choose columns for correlation heatmap",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
        )
        st.write("### Correlation Heatmap")
        plot_correlation_custom(df_all, chosen_corr_cols)

    # 3) Window Distribution Comparison
    if "Window Distribution Comparison" in selected_analyses:
        st.write("### Window Distribution Comparison")
        plot_distribution_comparison(df_all, data_window)

    # 4) Anomaly Detection (Window)
    if "Anomaly Detection (Window)" in selected_analyses:
        st.write("### Anomaly Detection (Current Window)")

        # We'll use a form so that the page doesn't refresh on every input change:
        with st.form("anomaly_form"):
            numeric_cols = data_window.select_dtypes(include=[np.number]).columns.tolist()

            # Default to some temperature columns (replace these with the actual column names you have)
            default_temp_cols = [c for c in numeric_cols if "temp_in_z0" in c or "temp_out_z0" in c]
            if not default_temp_cols:
                # fallback if those specific columns don't exist
                default_temp_cols = numeric_cols[:1]

            selected_anomaly_cols = st.multiselect(
                "Columns to analyze for anomalies (Z-score):",
                numeric_cols,
                default=default_temp_cols
            )
            z_threshold = st.slider("Z-score threshold:", 2.0, 5.0, 3.0, 0.1)

            # Only run detection & plotting after user clicks
            update_button = st.form_submit_button("Update Anomalies")

        # If user clicked update, run detection and show results
        if update_button and selected_anomaly_cols:
            anomalies_df = detect_anomalies_zscore(data_window, selected_anomaly_cols, threshold=z_threshold)
            anomaly_rate = anomalies_df['is_anomaly'].mean()
            st.write(f"**Anomaly Rate (Window)**: {anomaly_rate*100:.2f}%")

            # Show only anomalous rows (up to 20)
            anomaly_rows = anomalies_df[anomalies_df['is_anomaly'] == True]
            if not anomaly_rows.empty:
                st.write("**Anomalous rows (up to 20 displayed)**:")
                st.dataframe(anomaly_rows.head(20))
            else:
                st.write("No anomalies found with current threshold and columns.")

            st.write("**Time-Series Plots**:")
            # Plot each selected column with anomalies
            for col in selected_anomaly_cols:
                plot_anomalies_time_series(anomalies_df, x_col='timestamp', y_col=col)

    # 5) Feature Importances
    if "Feature Importances" in selected_analyses and model is not None:
        st.write("### Feature Importances (Model)")
        plot_feature_importances(model, feature_columns)

###############################################################################
# Fragment 2a: "Streaming" version => run_every=10
###############################################################################
@fragment(run_every=10)
def predictions_tab_streaming(df_all: pd.DataFrame):
    """
    This fragment re-runs every 10 seconds.
    We only increment st.session_state.current_index and do "streaming logic" here.
    """

    st.subheader("Predictions - Streaming (every 10s)")

    # If we hit end of dataset, auto-stop
    if st.session_state.current_index >= len(df_all):
        st.warning("Reached the end of the dataset.")
        st.session_state.streaming = False  # Turn off streaming
        return

    data_window_size = st.session_state.get("demo_window_size", 3)

    data_window = get_window_data(df_all, st.session_state.current_index, data_window_size)
    model = st.session_state.get("model", None)
    data_with_preds = predict(model, data_window)

    st.write("Last few predictions in the current window:")
    st.dataframe(data_with_preds.tail(5))
    plot_timeseries_with_prediction_interactive(data_with_preds)

    # Advance index by 1 each partial re-run
    st.session_state.current_index += 1

###############################################################################
# Fragment 2b: "Paused" version => run_every=None
###############################################################################
@fragment(run_every=None)
def predictions_tab_paused(df_all: pd.DataFrame):
    """
    This fragment is never auto re-run. It just shows "Paused" state
    whenever streaming=False. That means st.session_state.current_index doesn't change.
    """

    st.subheader("Predictions - Paused")

    end_idx = min(st.session_state.current_index, len(df_all) - 1)
    model = st.session_state.get("model", None)
    data_window_size = st.session_state.get("demo_window_size", 3)
    data_window = get_window_data(df_all, end_idx, data_window_size)
    data_with_preds = predict(model, data_window)

    st.write("Last few predictions in the current window:")
    if not data_with_preds.empty:
        st.dataframe(data_with_preds.tail(5))
    else:
        st.write("No data available in the current window.")

    plot_timeseries_with_prediction_interactive(data_with_preds)

###############################################################################
# "Unified" UI to switch between fragments
###############################################################################
def predictions_tab_controller(df_all: pd.DataFrame):
    """
    One function that shows Start/Stop buttons, then calls either
    predictions_tab_streaming or predictions_tab_paused
    depending on st.session_state.streaming.
    """

    st.subheader("Predictions and Real-Time Stream")

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

    if st.session_state.streaming:
        st.write("**Streaming is ON.**")
        predictions_tab_streaming(df_all)
    else:
        st.write("**Streaming is paused.**")
        predictions_tab_paused(df_all)

###############################################################################
# Main App
###############################################################################
def main():
    st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")

    # Hide some stale elements
    st.markdown(
        """
        <style>
        [data-stale="true"],
        .st-fk,
        .st-fl {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image("res/Miningful_NoBG_WhiteText.png", width=150)
    st.title("Miningful Predictive Maintenance Demo")

    # Sidebar config
    st.sidebar.header("⚙️ Configuration")
    demo_window_size = st.sidebar.slider("Data Window Size (days):", 3, 7, 3)
    st.sidebar.write("Polling interval: 10s")  # fixed

    st.session_state.demo_window_size = demo_window_size

    # Load data once
    if "stream_data" not in st.session_state:
        st.session_state.stream_data = load_data_remote()
    df_all = st.session_state.stream_data

    # Train model once
    if "model" not in st.session_state:
        with st.spinner("Training model..."):
            model, mse = train_model(df_all)
            st.session_state.model = model
            st.session_state.model_mse = mse

    if st.session_state.model is not None:
        st.success("Model training complete!")
        st.metric("Mean Squared Error", round(st.session_state.model_mse, 2))
    else:
        st.warning("Not enough data to train the model.")

    # On first load, define initial window index
    if "initial_window_set" not in st.session_state:
        dataset_start = df_all['timestamp'].min()
        initial_end_time = dataset_start + timedelta(days=demo_window_size)
        idx = find_index_for_time(df_all, initial_end_time)
        st.session_state.current_index = idx
        st.session_state.initial_window_set = True

    # Ensure we have a streaming flag
    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    # Feature columns (same as used in model training)
    feature_columns = [
        'raw_in_left', 'raw_in_right', 'raw_out_left', 'raw_out_right',
        'paperwidth_in', 'paperwidth_out', 'temp_in_z0', 'temp_out_z0',
        'temp_in_z1', 'temp_out_z1', 'temp_in_z2', 'temp_out_z2',
        'temp_in_z3', 'temp_out_z3', 'temp_in_z4', 'temp_out_z4',
        'temp_in_z5', 'temp_out_z5', 'temp_in_z6', 'temp_out_z6',
        'temp_in_z7', 'temp_out_z7', 'temp_in_z8', 'temp_out_z8',
        'temp_in_z9', 'temp_out_z9'
    ]

    # Tabs: Predictions vs. Data Exploration
    tab1, tab2 = st.tabs(["Predictions", "Data Exploration"])

    with tab1:
        predictions_tab_controller(df_all)

    with tab2:
        data_exploration_tab(df_all, st.session_state.model, feature_columns)

if __name__ == "__main__":
    main()
