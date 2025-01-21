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
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
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
# Enhanced Anomaly Detection
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
# Feature Importance Utilities
###############################################################################
def plot_feature_importances(model, feature_names):
    """Simple bar chart of feature importances."""
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

def get_top_n_features(model, feature_names, n=8):
    """Returns the top n features based on feature_importances_, sorted descending."""
    if not hasattr(model, 'feature_importances_'):
        return []
    importances = model.feature_importances_
    indices_desc = np.argsort(importances)[::-1]  # descending order
    top_indices = indices_desc[:n]
    return [feature_names[i] for i in top_indices]

###############################################################################
# Helper for plotting 8 separate charts
###############################################################################
def plot_separate_feature_charts(df: pd.DataFrame, features: list):
    """
    Plots each feature separately in a 4x2 grid (8 plots total).
    """
    # We'll place them in pairs (2 columns in each row).
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(features):
                feature = features[i + j]
                fig = px.line(
                    df,
                    x='timestamp',
                    y=feature,
                    title=f"{feature}"
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                cols[j].plotly_chart(fig, use_container_width=True)

###############################################################################
# Fragment 1: Data Exploration
###############################################################################
@fragment
def data_exploration_tab(df_all: pd.DataFrame, model, feature_columns):
    st.subheader("Data Exploration")

    end_idx = st.session_state.current_index
    data_window = get_window_data(df_all, end_idx, st.session_state.get("demo_window_size", 3))

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

        with st.form("anomaly_form"):
            numeric_cols = data_window.select_dtypes(include=[np.number]).columns.tolist()

            default_temp_cols = [c for c in numeric_cols if "temp_in_z0" in c or "temp_out_z0" in c]
            if not default_temp_cols:
                default_temp_cols = numeric_cols[:1]

            selected_anomaly_cols = st.multiselect(
                "Columns to analyze for anomalies (Z-score):",
                numeric_cols,
                default=default_temp_cols
            )
            z_threshold = st.slider("Z-score threshold:", 2.0, 5.0, 3.0, 0.1)
            update_button = st.form_submit_button("Update Anomalies")

        if update_button and selected_anomaly_cols:
            anomalies_df = detect_anomalies_zscore(data_window, selected_anomaly_cols, threshold=z_threshold)
            anomaly_rate = anomalies_df['is_anomaly'].mean()
            st.write(f"**Anomaly Rate (Window)**: {anomaly_rate*100:.2f}%")

            anomaly_rows = anomalies_df[anomalies_df['is_anomaly'] == True]
            if not anomaly_rows.empty:
                st.write("**Anomalous rows (up to 20 displayed)**:")
                st.dataframe(anomaly_rows.head(20))
            else:
                st.write("No anomalies found with current threshold and columns.")

            st.write("**Time-Series Plots**:")
            for col in selected_anomaly_cols:
                plot_anomalies_time_series(anomalies_df, x_col='timestamp', y_col=col)

    # 5) Feature Importances
    if "Feature Importances" in selected_analyses and model is not None:
        st.write("### Feature Importances (Model)")
        plot_feature_importances(model, feature_columns)

###############################################################################
# Fragment 2a: "Streaming" Predictions => run_every=10
###############################################################################
@fragment(run_every=10)
def predictions_tab_streaming(df_all: pd.DataFrame):
    if st.session_state.current_index >= len(df_all):
        st.warning("Reached the end of the dataset.")
        st.session_state.streaming = False
        return

    data_window_size = st.session_state.get("demo_window_size", 3)
    data_window = get_window_data(df_all, st.session_state.current_index, data_window_size)
    model = st.session_state.get("model", None)
    data_with_preds = predict(model, data_window)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Predictions - Streaming (every 10s)")
        st.dataframe(data_with_preds.tail(5), use_container_width=True)

    with col_right:
        plot_timeseries_with_prediction_interactive(data_with_preds)

    st.session_state.current_index += 1

###############################################################################
# Fragment 2b: "Paused" Predictions => run_every=None
###############################################################################
@fragment(run_every=None)
def predictions_tab_paused(df_all: pd.DataFrame):
    end_idx = min(st.session_state.current_index, len(df_all) - 1)
    model = st.session_state.get("model", None)
    data_window_size = st.session_state.get("demo_window_size", 3)
    data_window = get_window_data(df_all, end_idx, data_window_size)
    data_with_preds = predict(model, data_window)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Predictions - Paused")
        if not data_with_preds.empty:
            st.dataframe(data_with_preds.tail(5), use_container_width=True)
        else:
            st.write("No data available in the current window.")

    with col_right:
        plot_timeseries_with_prediction_interactive(data_with_preds)

###############################################################################
# Fragment 3a: Key Variables Streaming => run_every=10
###############################################################################
@fragment(run_every=10)
def key_variables_tab_streaming(df_all: pd.DataFrame):
    """
    Shows 8 separate line plots of the top 8 most important features
    over the last 50 rows *relative to the current index*.
    This re-runs every 10 seconds if streaming is ON.
    """
    top_features = st.session_state.get("top_features", [])
    if not top_features:
        st.write("No top features found. Please click 'Start Streaming' to compute.")
        return

    ### CHANGED HERE ###
    # Get a sliding window of 50 rows ending at current_index
    current_idx = st.session_state.current_index
    start_idx = max(0, current_idx - 49)
    df_last_50 = df_all.iloc[start_idx : current_idx + 1]
    st.markdown("#### Key Variables - Streaming (sliding window of last 50 rows)")

    # Plot each feature separately
    plot_separate_feature_charts(df_last_50, top_features)

###############################################################################
# Fragment 3b: Key Variables Paused => run_every=None
###############################################################################
@fragment(run_every=None)
def key_variables_tab_paused(df_all: pd.DataFrame):
    """
    Shows 8 separate line plots of the top 8 most important features
    over the last 50 rows *relative to the current index*, but does not auto-refresh.
    """
    top_features = st.session_state.get("top_features", [])
    if not top_features:
        st.write("No top features found. Streaming not started or model unavailable.")
        return

    ### CHANGED HERE ###
    # Same sliding window logic, but no auto increment
    current_idx = st.session_state.current_index
    start_idx = max(0, current_idx - 49)
    df_last_50 = df_all.iloc[start_idx : current_idx + 1]

    st.markdown("#### Key Variables - Paused (sliding window of last 50 rows)")
    plot_separate_feature_charts(df_last_50, top_features)

###############################################################################
# Controllers
###############################################################################
def predictions_tab_controller(df_all: pd.DataFrame):
    """
    Start/Stop streaming for predictions; calls the appropriate fragment.
    Also triggers feature-importance calculation if not done yet.
    """
    st.subheader("Predictions and Real-Time Stream")

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.streaming:
            if st.button("▶️ Start Streaming"):
                # When streaming starts, compute top 8 features if not already computed
                if "top_features" not in st.session_state:
                    model = st.session_state.get("model", None)
                    if model is not None:
                        feature_cols = st.session_state.get("feature_columns", [])
                        st.session_state.top_features = get_top_n_features(model, feature_cols, n=8)
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

def key_variables_tab_controller(df_all: pd.DataFrame):
    """
    Shows the top-8-features plots in either streaming or paused mode,
    updating at the same frequency as predictions, but always for the last 50 rows
    up to the current index.
    """
    st.subheader("Key Variables (Top 8 Features)")
    if st.session_state.streaming:
        st.write("**Streaming is ON.**")
        key_variables_tab_streaming(df_all)
    else:
        st.write("**Streaming is paused.**")
        key_variables_tab_paused(df_all)

###############################################################################
# Main App
###############################################################################
def main():
    st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")

    # -- Reduce extra spacing with custom CSS --
    st.markdown(
        """
        <style>
        h1, h2, h3, h4, h5, h6 {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .stMetric {
            padding: 0rem;
            margin: 0rem;
        }
        .element-container .stPlotlyChart {
            margin: 0rem 0rem;
            padding: 0rem 0rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image("res/Miningful_NoBG_WhiteText.png", width=120)
    st.markdown("### Miningful Predictive Maintenance Demo")

    # Sidebar config
    st.sidebar.header("⚙️ Configuration")
    demo_window_size = st.sidebar.slider("Data Window Size (days):", 3, 7, 3)
    st.sidebar.write("Polling interval: 10s (fixed)")

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
        mse_value = round(st.session_state.model_mse, 2)
        st.markdown(
            f"<span style='color:green;font-weight:bold;'>Model training complete!</span> MSE: **{mse_value}**",
            unsafe_allow_html=True
        )
    else:
        st.warning("Not enough data to train the model.")

    # Set initial window/index if not done yet
    if "initial_window_set" not in st.session_state:
        dataset_start = df_all['timestamp'].min()
        initial_end_time = dataset_start + timedelta(days=demo_window_size)
        idx = find_index_for_time(df_all, initial_end_time)
        st.session_state.current_index = idx
        st.session_state.initial_window_set = True

    # Ensure streaming state
    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    # Store feature columns for top-features logic
    feature_columns = [
        'raw_in_left', 'raw_in_right', 'raw_out_left', 'raw_out_right',
        'paperwidth_in', 'paperwidth_out', 'temp_in_z0', 'temp_out_z0',
        'temp_in_z1', 'temp_out_z1', 'temp_in_z2', 'temp_out_z2',
        'temp_in_z3', 'temp_out_z3', 'temp_in_z4', 'temp_out_z4',
        'temp_in_z5', 'temp_out_z5', 'temp_in_z6', 'temp_out_z6',
        'temp_in_z7', 'temp_out_z7', 'temp_in_z8', 'temp_out_z8',
        'temp_in_z9', 'temp_out_z9'
    ]
    st.session_state.feature_columns = feature_columns

    # Tabs: Predictions, Data Exploration, Key Variables
    tab1, tab2, tab3 = st.tabs(["Predictions", "Data Exploration", "Key Variables"])

    with tab1:
        predictions_tab_controller(df_all)

    with tab2:
        data_exploration_tab(df_all, st.session_state.model, feature_columns)

    with tab3:
        key_variables_tab_controller(df_all)

if __name__ == "__main__":
    main()
