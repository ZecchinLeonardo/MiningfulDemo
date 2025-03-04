import streamlit as st
from streamlit.runtime.fragment import fragment
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

import os
import boto3
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

###############################################################################
# Fixed Start Time Constant (for sliding window)
###############################################################################
FIXED_START_TIME = datetime(2023, 2, 17, 9, 23)  # Fixed start: Feb 17 at 9:23

###############################################################################
# 1) AWS Credentials and S3 client
###############################################################################
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

###############################################################################
# 2) Data Loading and Model Training
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
# NEW: Sliding Window Helper (Fixed Start + Evolving Offset)
###############################################################################
def get_sliding_window_data(df: pd.DataFrame, offset: timedelta, window_duration_hours: float) -> pd.DataFrame:
    """
    Returns the data between:
      window_start = FIXED_START_TIME + offset 
      window_end   = window_start + (window_duration_hours)
    
    If the timestamp column is timezone-aware, it is converted to naive.
    """
    window_start = FIXED_START_TIME + offset
    window_end = window_start + timedelta(hours=window_duration_hours)
    if isinstance(df['timestamp'].dtype, pd.DatetimeTZDtype):
        ts = df['timestamp'].dt.tz_convert(None)
    else:
        ts = df['timestamp']
    return df[(ts >= window_start) & (ts <= window_end)]

###############################################################################
# 3) Prediction & Plotting Utilities
###############################################################################
def predict(model, data_window: pd.DataFrame) -> pd.DataFrame:
    """Use the model to predict 'predicted_moisture' from feature columns."""
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
        title="Actual vs. Predicted Moisture Over Time",
        color_discrete_sequence=["royalblue", "tomato"]
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 4) Data Exploration & Visualization
###############################################################################
def plot_distributions_custom(df: pd.DataFrame, columns_to_plot: list):
    if not columns_to_plot:
        st.write("No columns selected.")
        return
    num_cols = len(columns_to_plot)
    fig = make_subplots(rows=1, cols=num_cols, subplot_titles=columns_to_plot)
    for i, col in enumerate(columns_to_plot, start=1):
        if df[col].dropna().empty:
            fig.add_annotation(
                text=f"No data for {col}",
                xref=f"x{i} domain", yref=f"y{i} domain",
                showarrow=False, row=1, col=i
            )
        else:
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    histnorm='density',
                    name=col,
                    marker_color='royalblue',
                    opacity=0.75,
                    showlegend=False
                ),
                row=1, col=i
            )
    fig.update_layout(barmode='overlay', height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_custom(df: pd.DataFrame, columns_to_plot: list):
    if not columns_to_plot:
        st.write("No columns selected for correlation.")
        return
    sub_df = df[columns_to_plot].dropna()
    if sub_df.shape[1] < 2:
        st.write("Not enough columns selected for correlation heatmap.")
        return
    corr = sub_df.corr()
    custom_colorscale = [[0, 'royalblue'], [0.5, 'white'], [1, 'tomato']]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=custom_colorscale
    ))
    fig.update_layout(title="Correlation Heatmap (Selected Columns)", height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_distribution_comparison_2x4(df_all: pd.DataFrame, df_window: pd.DataFrame, columns: list):
    # Guard against an empty columns list.
    if not columns:
        st.write("No columns selected for distribution comparison.")
        return
    
    n = len(columns)
    rows = math.ceil(n / 4)
    fig = make_subplots(rows=rows, cols=4, subplot_titles=columns)
    for i, col in enumerate(columns, start=1):
        row = math.ceil(i / 4)
        col_pos = i - (row - 1) * 4
        
        if col in df_all.columns and pd.api.types.is_numeric_dtype(df_all[col]):
            fig.add_trace(
                go.Histogram(
                    x=df_all[col].dropna(),
                    histnorm='density',
                    name='Overall',
                    marker_color='royalblue',
                    opacity=0.6,
                    showlegend=(i == 1)
                ),
                row=row, col=col_pos
            )
        if col in df_window.columns and pd.api.types.is_numeric_dtype(df_window[col]):
            fig.add_trace(
                go.Histogram(
                    x=df_window[col].dropna(),
                    histnorm='density',
                    name='Window',
                    marker_color='tomato',
                    opacity=0.6,
                    showlegend=(i == 1)
                ),
                row=row, col=col_pos
            )
        fig.update_xaxes(title_text=col, row=row, col=col_pos)
    fig.update_layout(barmode='overlay', height=400 * rows)
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importances(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        st.write("Model does not have feature importances.")
        return
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    fig = px.bar(
        x=sorted_importances,
        y=sorted_features,
        orientation='h',
        labels={'x': 'Importance', 'y': 'Features'},
        title="Feature Importances (RandomForest)",
        color_discrete_sequence=['royalblue']
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
def plot_anomaly_detection_graph(df: pd.DataFrame, anomaly_cols: list):
    """
    For each column in anomaly_cols that has at least one anomaly in the dataframe,
    plot the time series with normal values as a line (in a distinct color)
    and anomalies highlighted as markers (using the same distinct color with a black outline).
    Only variables with anomalies are plotted.
    """
    fig = go.Figure()
    omitted = []
    colors = px.colors.qualitative.Plotly  # distinct colors for each variable
    for i, col in enumerate(anomaly_cols):
        if not df[df[f"{col}_anomaly"]].empty:
            color = colors[i % len(colors)]
            normal = df[~df[f"{col}_anomaly"]]
            anomalous = df[df[f"{col}_anomaly"]]
            fig.add_trace(
                go.Scatter(
                    x=normal['timestamp'], y=normal[col],
                    mode='lines',
                    name=f"{col} Normal",
                    line=dict(color=color)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=anomalous['timestamp'], y=anomalous[col],
                    mode='markers',
                    name=f"{col} Anomaly",
                    marker=dict(color=color, size=10, symbol="x", line=dict(width=1, color='black'))
                )
            )
        else:
            omitted.append(col)
    if omitted:
        st.warning(f"The following variables do not have anomalies and are not displayed: {', '.join(omitted)}")
    if fig.data:
        fig.update_layout(title="Anomaly Detection Over Time", height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies detected in the selected variables.")

def plot_features_2x4_subplots_anomaly(df: pd.DataFrame, features: list):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    fig = make_subplots(rows=2, cols=4, subplot_titles=features)
    row_col_map = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4)]
    for i, feat in enumerate(features):
        row, col = row_col_map[i]
        anom_col = f"{feat}_anomaly"
        mask_anom = df.get(anom_col, False)
        # Plot the normal values in royal blue.
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[feat],
                mode="lines",
                name=f"{feat} (all)",
                line=dict(color="royalblue")
            ),
            row=row, col=col
        )
        # Plot anomalies in tomato with a black outline.
        df_anom = df[mask_anom]
        if not df_anom.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_anom["timestamp"],
                    y=df_anom[feat],
                    mode="markers",
                    name=f"{feat} (anomaly)",
                    marker=dict(color="tomato", size=10, symbol="x", line=dict(color="black", width=1))
                ),
                row=row, col=col
            )
    for axis_name in fig.layout:
        if axis_name.startswith("xaxis"):
            fig.layout[axis_name].rangebreaks = [dict(pattern="day of week", bounds=["sat", "sun"])]
            fig.layout[axis_name].type = "date"
    fig.update_layout(
        autosize=False,
        width=1200,
        height=400,
        showlegend=False,
        title_text="Key Variables (Sliding Window, Skip Weekends)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=False)


###############################################################################
# 5) Anomaly Detection: Z-Score on the Top 8 Features
###############################################################################
def detect_anomalies_for_features(df: pd.DataFrame, features: list, z_threshold=3.0) -> pd.DataFrame:
    """
    Returns a copy of df with:
      - For each feature in `features`, a column named <feature>_anomaly which is True if
        the absolute z-score exceeds z_threshold.
      - A column any_anomaly = True if any feature is anomalous in that row.
    """
    df_out = df.copy()
    for feat in features:
        if feat in df_out.columns and pd.api.types.is_numeric_dtype(df_out[feat]):
            mean_ = df_out[feat].mean()
            std_ = df_out[feat].std()
            if std_ == 0:
                df_out[f"{feat}_anomaly"] = False
            else:
                zscores = (df_out[feat] - mean_) / std_
                df_out[f"{feat}_anomaly"] = (zscores.abs() > z_threshold)
    anomaly_cols = [f"{f}_anomaly" for f in features if f"{f}_anomaly" in df_out.columns]
    if anomaly_cols:
        df_out["any_anomaly"] = df_out[anomaly_cols].any(axis=1)
    else:
        df_out["any_anomaly"] = False
    return df_out

###############################################################################
# 6b) Compute Performance Metrics
###############################################################################
def compute_performance_metrics(df: pd.DataFrame, target_col: str = "moisture_in_z0", pred_col: str = "predicted_moisture") -> dict:
    df_valid = df.dropna(subset=[target_col, pred_col])
    if df_valid.empty:
        return {"mse": None, "mae": None, "mape": None, "r2": None}
    y_true = df_valid[target_col]
    y_pred = df_valid[pred_col]
    mse_val = mean_squared_error(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    r2_val  = r2_score(y_true, y_pred)
    # Compute MAPE, using np.where to avoid division by zero issues.
    mape_val = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    return {"mse": mse_val, "mae": mae_val, "mape": mape_val, "r2": r2_val}

def get_top_n_features(model, feature_names, n=8):
    """Return top-n features based on feature_importances_, descending."""
    if not hasattr(model, 'feature_importances_'):
        return []
    importances = model.feature_importances_
    indices_desc = np.argsort(importances)[::-1]
    top_indices = indices_desc[:n]
    return [feature_names[i] for i in top_indices]

###############################################################################
# 6c) Key Variables Plot: (unchanged here, updated above)
###############################################################################
# (Using plot_features_2x4_subplots_anomaly defined above)

###############################################################################
# 8) Data Exploration Fragment (Modified)
###############################################################################
@fragment
def data_exploration_tab(df_all: pd.DataFrame, model, feature_columns):

    # -- Auto-compute top features if not available --
    if "top_features" not in st.session_state or not st.session_state.top_features:
        if model is not None and "feature_columns" in st.session_state:
            st.session_state.top_features = get_top_n_features(
                model, st.session_state.feature_columns, n=8
            )

    # -- Get sliding window data --
    window_duration = st.session_state.window_duration
    current_offset = st.session_state.get("current_offset", timedelta(0))
    data_window = get_sliding_window_data(df_all, current_offset, window_duration)

    # -- Default top features if available --
    default_top_features = st.session_state.get("top_features", None)

    # -- Analysis options for user selection --
    analysis_options = [
        "Correlation Heatmap (Full Dataset)",
        "Window Distribution Comparison",
        "Anomaly Detection (Window)",
        "Feature Importances"
    ]
    selected_analyses = st.multiselect(
        "Select Analyses to Display",
        analysis_options,
        default=[
            "Window Distribution Comparison",
            "Anomaly Detection (Window)",
            "Feature Importances",
            "Correlation Heatmap (Full Dataset)"
        ]
    )

    # ─────────────────────────────────────────────────────────
    #  1) CORRELATION HEATMAP & FEATURE IMPORTANCES SIDE BY SIDE
    # ─────────────────────────────────────────────────────────
    correlation_selected = "Correlation Heatmap (Full Dataset)" in selected_analyses
    feature_importances_selected = "Feature Importances" in selected_analyses

    # Both selected → two columns
    if correlation_selected and feature_importances_selected:
        left_col, right_col = st.columns(2)

        # Correlation Heatmap in Left Column
        with left_col:
            numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
            default_corr = (
                default_top_features 
                if default_top_features and len(default_top_features) >= 2 
                else (numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
            )
            chosen_corr_cols = st.multiselect(
                "Columns for correlation heatmap",
                numeric_cols,
                default=default_corr
            )
            st.write("#### Correlation Heatmap")
            plot_correlation_custom(df_all, chosen_corr_cols)

        # Feature Importances in Right Column
        with right_col:
            st.write("#### Feature Importances (Model)")
            plot_feature_importances(model, feature_columns)

    # If *only* correlation is selected
    elif correlation_selected:
        numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        default_corr = (
            default_top_features 
            if default_top_features and len(default_top_features) >= 2 
            else (numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
        )
        chosen_corr_cols = st.multiselect(
            "Columns for correlation heatmap",
            numeric_cols,
            default=default_corr
        )
        st.write("#### Correlation Heatmap")
        plot_correlation_custom(df_all, chosen_corr_cols)

    # If *only* feature importances is selected
    elif feature_importances_selected:
        st.write("#### Feature Importances (Model)")
        plot_feature_importances(model, feature_columns)

    # ─────────────────────────────────────────────────────────
    #  2) WINDOW DISTRIBUTION COMPARISON (FULL WIDTH)
    # ─────────────────────────────────────────────────────────
    if "Window Distribution Comparison" in selected_analyses:
        st.write("### Window Distribution Comparison")
        default_dist_cols = (
            default_top_features 
            if default_top_features and len(default_top_features) >= 8 
            else ['raw_in_left', 'raw_in_right', 'raw_out_left', 'raw_out_right']
        )
        selected_cols = st.multiselect(
            "Select columns for distribution comparison",
            df_all.select_dtypes(include=[np.number]).columns.tolist(),
            default=default_dist_cols
        )
        plot_distribution_comparison_2x4(df_all, data_window, selected_cols)

    # ─────────────────────────────────────────────────────────
    #  3) ANOMALY DETECTION (WINDOW) - BELOW
    # ─────────────────────────────────────────────────────────
    if "Anomaly Detection (Window)" in selected_analyses:
        st.write("### Anomaly Detection (Current Window)")
        with st.form("anomaly_form"):
            numeric_cols = data_window.select_dtypes(include=[np.number]).columns.tolist()
            default_anomaly = (
                default_top_features 
                if default_top_features and len(default_top_features) > 0 
                else (numeric_cols[:1] if numeric_cols else [])
            )
            selected_anomaly_cols = st.multiselect(
                "Columns to analyze for anomalies (Z-score):",
                numeric_cols,
                default=default_anomaly
            )
            z_threshold = st.slider("Z-score threshold:", 2.0, 5.0, 3.0, 0.1)
            update_button = st.form_submit_button("Update Anomalies")

        if update_button and selected_anomaly_cols:
            anomalies_df = detect_anomalies_for_features(data_window, selected_anomaly_cols, z_threshold=z_threshold)
            anomaly_rate = anomalies_df['any_anomaly'].mean()
            st.write(f"**Anomaly Rate (Window)**: {anomaly_rate*100:.2f}%")
            anomaly_rows = anomalies_df[anomalies_df['any_anomaly'] == True]
            if not anomaly_rows.empty:
                st.write("**Anomalous rows (up to 20 displayed)**:")
                st.dataframe(anomaly_rows.head(20))
            else:
                st.write("No anomalies found with current threshold and columns.")
            # Produce a graph for anomaly detection.
            plot_anomaly_detection_graph(anomalies_df, selected_anomaly_cols)

###############################################################################
# 9) Predictions Tab (Streaming & Paused, Sliding Window)
###############################################################################
@fragment(run_every=3)
def predictions_tab_streaming(df_all: pd.DataFrame):
    """Re-runs every 10 seconds; the sliding window moves forward by a fixed increment."""
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True).dt.tz_convert(None)
    window_duration = st.session_state.window_duration
    if "current_offset" not in st.session_state:
        st.session_state.current_offset = timedelta(0)
    data_window = get_sliding_window_data(df_all, st.session_state.current_offset, window_duration)
    model = st.session_state.get("model", None)
    data_with_preds = predict(model, data_window)
    top_features = st.session_state.get("top_features", [])
    df_ana = detect_anomalies_for_features(data_with_preds, top_features, z_threshold=3.0)
    st.session_state["df_main_anomalies"] = df_ana

    def highlight_anomaly_row(row):
        if row.get("any_anomaly", False):
            return ["background-color: yellow"] * len(row)
        else:
            return ["" for _ in row]
    
    df_last5 = df_ana.sort_values('timestamp').tail(5)
    # Reorder columns: keep 'timestamp' first, then 'moisture_in_z0' and 'predicted_moisture' as second and third.
    cols = list(df_last5.columns)
    desired_order = []
    if 'timestamp' in cols:
        desired_order.append('timestamp')
    if 'moisture_in_z0' in cols:
        desired_order.append('moisture_in_z0')
    if 'predicted_moisture' in cols:
        desired_order.append('predicted_moisture')
    for col in cols:
        if col not in desired_order:
            desired_order.append(col)
    df_last5 = df_last5[desired_order]

    df_styled = df_last5.style.apply(highlight_anomaly_row, axis=1)
    
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("#### Predictions - Streaming (Sliding Window)")
        st.write(df_styled)
        anomaly_rows = df_last5[df_last5["any_anomaly"] == True]
        if not anomaly_rows.empty:
            st.write("**Anomaly columns (within top 8)** for these rows:")
            for idx, row in anomaly_rows.iterrows():
                anom_cols = [f for f in top_features if row.get(f"{f}_anomaly", False)]
                if anom_cols:
                    st.write(f"Row {idx}: {', '.join(anom_cols)}")
        else:
            st.write("No anomalies in the displayed records (top 8 features).")
    with col_right:
        plot_timeseries_with_prediction_interactive(df_ana)
    
    # NEW: Add Key Variables (Sliding Window) below the predictions.
    st.markdown("### Key Variables (Sliding Window)")
    if not top_features:
        st.warning("Top features are not computed yet. Please start streaming to compute top features.")
    else:
        plot_features_2x4_subplots_anomaly(df_ana, top_features)
    
    st.session_state.current_offset += timedelta(seconds=10)

@fragment(run_every=None)
def predictions_tab_paused(df_all: pd.DataFrame):
    """Paused state: uses the current sliding window without advancing the offset."""
    window_duration = st.session_state.window_duration
    current_offset = st.session_state.get("current_offset", timedelta(0))
    model = st.session_state.get("model", None)
    data_window = get_sliding_window_data(df_all, current_offset, window_duration)
    data_with_preds = predict(model, data_window)
    top_features = st.session_state.get("top_features", [])
    df_ana = detect_anomalies_for_features(data_with_preds, top_features, z_threshold=3.0)
    
    def highlight_anomaly_row(row):
        if row.get("any_anomaly", False):
            return ["background-color: yellow"] * len(row)
        else:
            return ["" for _ in row]
    
    df_last5 = df_ana.sort_values('timestamp').tail(5)
    cols = list(df_last5.columns)
    desired_order = []
    if 'timestamp' in cols:
        desired_order.append('timestamp')
    if 'moisture_in_z0' in cols:
        desired_order.append('moisture_in_z0')
    if 'predicted_moisture' in cols:
        desired_order.append('predicted_moisture')
    for col in cols:
        if col not in desired_order:
            desired_order.append(col)
    df_last5 = df_last5[desired_order]

    df_styled = df_last5.style.apply(highlight_anomaly_row, axis=1)
    
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("#### Predictions - Paused (Sliding Window)")
        st.write(df_styled)
        anomaly_rows = df_last5[df_last5["any_anomaly"] == True]
        if not anomaly_rows.empty:
            st.write("**Anomaly columns (within top 8)**:")
            for idx, row in anomaly_rows.iterrows():
                anom_cols = [f for f in top_features if row.get(f"{f}_anomaly", False)]
                if anom_cols:
                    st.write(f"Row {idx}: {', '.join(anom_cols)}")
        else:
            st.write("No anomalies in the displayed records (top 8 features).")
    with col_right:
        plot_timeseries_with_prediction_interactive(df_ana)
    
    st.markdown("### Key Variables (Sliding Window)")
    if not top_features:
        st.warning("Top features are not computed yet. Please start streaming to compute top features.")
    else:
        plot_features_2x4_subplots_anomaly(df_ana, top_features)

###############################################################################
# 12) Model Performance Tab (Now Includes Key Variables)
###############################################################################
@fragment(run_every=None)
def model_performance_tab(df_all: pd.DataFrame, model, feature_columns):
    if model is None:
        st.warning("No trained model available. Please train the model first.")
        return
    df_all_preds = predict(model, df_all)
    df_all_preds.rename(
        columns={"predicted_moisture": "predicted_moisture_full"}, 
        inplace=True
    )
    full_metrics = compute_performance_metrics(
        df_all_preds, 
        target_col="moisture_in_z0", 
        pred_col="predicted_moisture_full"
    )
    window_duration = st.session_state.window_duration
    current_offset = st.session_state.get("current_offset", timedelta(0))
    data_window = get_sliding_window_data(df_all, current_offset, window_duration)
    df_window_preds = predict(model, data_window)
    window_metrics = compute_performance_metrics(
        df_window_preds, 
        target_col="moisture_in_z0", 
        pred_col="predicted_moisture"
    )
    
    col1, col2 = st.columns(2)
    
    # Full Dataset Metrics with hierarchical sizes and fixed colors:
    with col1:
        st.markdown("<h1>Full Dataset Metrics</h1>", unsafe_allow_html=True)
        if full_metrics["mse"] is None:
            st.write("No valid rows for full-dataset metric computation.")
        else:
            st.markdown(f"<h2 style='color:tomato;'>MAE (Full): {full_metrics['mae']:.4f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:tomato;'>MAPE (Full): {full_metrics['mape']:.2f}%</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:royalblue;'>MSE (Full): {full_metrics['mse']:.4f}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:royalblue;'>R² (Full): {full_metrics['r2']:.4f}</h3>", unsafe_allow_html=True)
    
    # Sliding Window Metrics with hierarchical sizes and fixed colors:
    with col2:
        st.markdown("<h1>Sliding Window Metrics</h1>", unsafe_allow_html=True)
        if window_metrics["mse"] is None:
            st.write("No valid rows in the sliding window for metric computation.")
        else:
            st.markdown(f"<h2 style='color:tomato;'>MAE (Window): {window_metrics['mae']:.4f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:tomato;'>MAPE (Window): {window_metrics['mape']:.2f}%</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:royalblue;'>MSE (Window): {window_metrics['mse']:.4f}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:royalblue;'>R² (Window): {window_metrics['r2']:.4f}</h3>", unsafe_allow_html=True)
    
    # Full Dataset: Actual vs. Predicted plot with fixed color sequence.
    st.markdown("### Full Dataset: Actual vs. Predicted")
    if "predicted_moisture_full" not in df_all_preds or df_all_preds["predicted_moisture_full"].dropna().empty:
        st.write("No valid predictions for full dataset.")
    else:
        fig_full = px.line(
            df_all_preds.dropna(subset=["predicted_moisture_full", "moisture_in_z0"]),
            x="timestamp",
            y=["moisture_in_z0", "predicted_moisture_full"],
            labels={"value": "Moisture", "variable": "Series", "timestamp": "Time"},
            title="Full Dataset: Actual vs. Predicted Moisture",
            color_discrete_sequence=["royalblue", "tomato"]
        )
        fig_full.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_full, use_container_width=True)
    
    # Sliding Window: Actual vs. Predicted plot with fixed color sequence.
    st.markdown("### Sliding Window: Actual vs. Predicted")
    if df_window_preds["predicted_moisture"].dropna().empty:
        st.write("No valid predictions for the sliding window.")
    else:
        fig_window = px.line(
            df_window_preds.dropna(subset=["predicted_moisture", "moisture_in_z0"]),
            x="timestamp",
            y=["moisture_in_z0", "predicted_moisture"],
            labels={"value": "Moisture", "variable": "Series", "timestamp": "Time"},
            title="Sliding Window: Actual vs. Predicted Moisture",
            color_discrete_sequence=["royalblue", "tomato"]
        )
        fig_window.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_window, use_container_width=True)
    
    if "retrained_on" in st.session_state:
        retrain_start, retrain_end = st.session_state.retrained_on
        st.markdown(f"**Model was retrained on data from {retrain_start} to {retrain_end}.**")

        
def predictions_tab_controller(df_all: pd.DataFrame):
    st.sidebar.header("⚙️ Configuration")
    window_duration = st.sidebar.slider(
        "Window Duration (hours):", 
        min_value=1.0, 
        max_value=2.0, 
        value=1.0, 
        step=0.5
    )
    st.session_state.window_duration = window_duration

    st.sidebar.header("🔄 Retrain Model on Custom Window")
    retrain_enabled = st.sidebar.checkbox("Enable Custom Training Window")
    if retrain_enabled:
        st.sidebar.write("### Select Custom Training Time Range")
        dataset_start = st.session_state.stream_data['timestamp'].min()
        dataset_end = st.session_state.stream_data['timestamp'].max()
        fixed_window_start_date = st.sidebar.date_input(
            "Start Date",
            value=dataset_start.date(),
            min_value=dataset_start.date(),
            max_value=dataset_end.date()
        )
        fixed_window_start_time = st.sidebar.time_input(
            "Start Time",
            value=dataset_start.time()
        )
        retrain_duration = st.sidebar.slider(
            "Duration (days):", 
            min_value=1, 
            max_value=30, 
            value=1, 
            step=1
        )
        fixed_window_end_datetime = datetime.combine(
            fixed_window_start_date, 
            fixed_window_start_time
        ) + timedelta(days=retrain_duration)
        fixed_window_end_date = st.sidebar.date_input(
            "End Date",
            value=fixed_window_end_datetime.date(),
            min_value=fixed_window_start_date,
            max_value=dataset_end.date()
        )
        fixed_window_end_time = st.sidebar.time_input(
            "End Time",
            value=fixed_window_end_datetime.time()
        )
        fixed_start_datetime = datetime.combine(fixed_window_start_date, fixed_window_start_time)
        fixed_end_datetime = datetime.combine(fixed_window_end_date, fixed_window_end_time)
        if fixed_end_datetime <= fixed_start_datetime:
            st.sidebar.error("End datetime must be after start datetime.")
        if st.sidebar.button("🔄 Retrain Model"):
            retrain_data = df_all[
                (df_all['timestamp'] >= fixed_start_datetime) & 
                (df_all['timestamp'] <= fixed_end_datetime)
            ].copy()
            if retrain_data.empty:
                st.sidebar.error("No data available in the selected window for retraining.")
            else:
                with st.spinner("Retraining model on the selected subset..."):
                    new_model, new_mse = train_model(retrain_data)
                    if new_model is not None:
                        st.session_state.model = new_model
                        st.session_state.model_mse = new_mse
                        st.session_state.retrained_on = (fixed_start_datetime, fixed_end_datetime)
                        st.success(f"Model retrained on data from {fixed_start_datetime} to {fixed_end_datetime}. MSE: {new_mse:.4f}")
                    else:
                        st.sidebar.error("Failed to retrain the model. Check if the selected window has sufficient data.")
    else:
        st.sidebar.write("Polling interval: 3s (fixed)")

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.get("streaming", False):
            if st.button("▶️ Start Streaming"):
                if "top_features" not in st.session_state or not st.session_state.top_features:
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
    if st.session_state.get("streaming", False):
        st.write("**Streaming is ON.**")
        predictions_tab_streaming(df_all)
    else:
        st.write("**Streaming is paused.**")
        predictions_tab_paused(df_all)

###############################################################################
# 13) Main App
###############################################################################
def main():
    st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
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
    if "stream_data" not in st.session_state:
        st.session_state.stream_data = load_data_remote()
    df_all = st.session_state.stream_data
    if "model" not in st.session_state:
        with st.spinner("Training model..."):
            model, mse = train_model(df_all)
            st.session_state.model = model
            st.session_state.model_mse = mse
    if st.session_state.model is not None:
        st.markdown("<span style='color:green;font-weight:bold;'>Model training complete!</span>", unsafe_allow_html=True)
    else:
        st.warning("Not enough data to train the model.")
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
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "current_offset" not in st.session_state:
        st.session_state.current_offset = timedelta(0)
    # Remove Key Variables tab; now we have only three tabs.
    tab1, tab2, tab3 = st.tabs(["Predictions", "Model Performance", "Data Exploration"])
    with tab1:
        predictions_tab_controller(df_all)
    with tab2:
        model_performance_tab(df_all, st.session_state.model, feature_columns)
    with tab3:
        data_exploration_tab(df_all, st.session_state.model, feature_columns)

if __name__ == "__main__":
    main()
