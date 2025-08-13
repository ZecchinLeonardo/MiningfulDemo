import streamlit as st
st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
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
import shap

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
    region_name='eu-central-1',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

###############################################################################
# 2) Data Loading and Model Training
###############################################################################
def load_data_remote():
    obj = s3.get_object(Bucket='miningfuldemo2', Key='datirs_SK.csv')
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data), sep=';', parse_dates=['to_timestamp'])
    df.rename(columns={'to_timestamp': 'timestamp'}, inplace=True)
    df.sort_values('timestamp', inplace=True, ignore_index=True)
    df = df[df['moisture_in_z0'] != 0]
    return df

def train_model(df: pd.DataFrame):
    """
    Trains a RandomForestRegressor using the *new* feature column names.
    We use 'moisture_in_z0' as our target, which remains unchanged in the new CSV.
    """
    # Updated feature column list
    feature_columns = [
        'shrink_raw_in_left', 'shrink_raw_in_right',
        'shrink_raw_out_left', 'shrink_raw_out_right',
        'paperwidth_in', 'paperwidth_out',

        'temperature_in_z0', 'temperature_out_z0',
        'temperature_in_z1', 'temperature_out_z1',
        'temperature_in_z2', 'temperature_out_z2',
        'temperature_in_z3', 'temperature_out_z3',
        'temperature_in_z4', 'temperature_out_z4',
        'temperature_in_z5', 'temperature_out_z5',
        'temperature_in_z6', 'temperature_out_z6',
        'temperature_in_z7', 'temperature_out_z7',
        'temperature_in_z8', 'temperature_out_z8',
        'temperature_in_z9', 'temperature_out_z9'
    ]
    target_column = 'moisture_in_z0'  # remains the same

    # Filter out rows missing *any* of the required columns (features + target)
    df_model = df.dropna(subset=feature_columns + [target_column])
    if df_model.empty:
        return None, None  # No data to train on

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
# 3) Sliding Window Helper
###############################################################################
def get_sliding_window_data(df: pd.DataFrame, offset: timedelta, window_duration_hours: float) -> pd.DataFrame:
    """
    Returns the data between:
      window_start = FIXED_START_TIME + offset 
      window_end   = window_start + window_duration (in hours)
    """
    window_start = FIXED_START_TIME + offset
    window_end = window_start + timedelta(hours=window_duration_hours)
    if isinstance(df['timestamp'].dtype, pd.DatetimeTZDtype):
        ts = df['timestamp'].dt.tz_convert(None)
    else:
        ts = df['timestamp']
    return df[(ts >= window_start) & (ts <= window_end)]

###############################################################################
# 4) Prediction & Plotting Utilities
###############################################################################
def predict(model, data_window: pd.DataFrame) -> pd.DataFrame:
    """
    Use the trained model to predict 'predicted_moisture' from the *new* feature columns.
    Force predicted moisture to 0 if the actual (moisture_in_z0) is 0.
    """
    if model is None or data_window.empty:
        return data_window

    # Same updated feature columns used in train_model
    feature_columns = [
        'shrink_raw_in_left', 'shrink_raw_in_right',
        'shrink_raw_out_left', 'shrink_raw_out_right',
        'paperwidth_in', 'paperwidth_out',

        'temperature_in_z0', 'temperature_out_z0',
        'temperature_in_z1', 'temperature_out_z1',
        'temperature_in_z2', 'temperature_out_z2',
        'temperature_in_z3', 'temperature_out_z3',
        'temperature_in_z4', 'temperature_out_z4',
        'temperature_in_z5', 'temperature_out_z5',
        'temperature_in_z6', 'temperature_out_z6',
        'temperature_in_z7', 'temperature_out_z7',
        'temperature_in_z8', 'temperature_out_z8',
        'temperature_in_z9', 'temperature_out_z9'
    ]
    target_column = 'moisture_in_z0'

    # Drop rows where *any* of the required features are missing
    X = data_window[feature_columns].dropna()
    if X.empty:
        return data_window

    preds = model.predict(X)
    data_window = data_window.copy()
    data_window['predicted_moisture'] = np.nan
    data_window.loc[X.index, 'predicted_moisture'] = preds

    # Force predicted moisture to 0 if the actual moisture_in_z0 is 0
    zero_mask = (data_window[target_column] == 0)
    data_window.loc[zero_mask, 'predicted_moisture'] = 0

    return data_window

def plot_timeseries_with_prediction_interactive(
    df: pd.DataFrame,
    model,
    feature_columns,
    time_col='timestamp',
    actual_col='moisture_in_z0',
    predicted_col='predicted_moisture',
    forecast=False,
    include_actual=False,
    overlay_df: pd.DataFrame | None = None,
    overlay_pred_col: str = 'predicted_moisture'
):
    if predicted_col not in df.columns or df[predicted_col].dropna().empty:
        st.write("No predictions to plot yet.")
        return

    base_df = df.copy()
    if isinstance(base_df[time_col].dtype, pd.DatetimeTZDtype):
        base_df[time_col] = base_df[time_col].dt.tz_convert(None)

    odf = None
    if overlay_df is not None:
        odf = overlay_df.copy()
        if time_col in odf.columns and isinstance(odf[time_col].dtype, pd.DatetimeTZDtype):
            odf[time_col] = odf[time_col].dt.tz_convert(None)

    df_filtered = base_df.dropna(subset=[predicted_col])
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_filtered[time_col],
            y=df_filtered[predicted_col],
            mode='lines',
            name="Predicted moisture",
            line=dict(color="tomato")
        )
    )

    if include_actual and actual_col in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered[time_col],
                y=df_filtered[actual_col],
                mode='lines',
                name="Actual moisture",
                line=dict(color="royalblue")
            )
        )

    if odf is not None and overlay_pred_col in odf.columns and not odf[overlay_pred_col].dropna().empty:
        odf2 = odf.dropna(subset=[overlay_pred_col]).copy()
        fig.add_trace(
            go.Scatter(
                x=odf2[time_col],
                y=odf2[overlay_pred_col],
                mode='lines',
                name="Prediction (adjusted)",
                line=dict(color="green", dash="dot")
            )
        )

    pred_window = df_filtered[predicted_col].dropna()
    pred_nonzero = pred_window[pred_window > 0]
    high_thr = pred_nonzero.quantile(0.90) if not pred_nonzero.empty else pred_window.quantile(0.90)
    low_thr = pred_nonzero.quantile(0.10) if not pred_nonzero.empty else pred_window.quantile(0.10)
    out_of_bounds = df_filtered[(df_filtered[predicted_col] > high_thr) | (df_filtered[predicted_col] < low_thr)]

    fig.add_hrect(y0=low_thr, y1=high_thr, fillcolor="rgba(255,255,255,0.04)", line_width=0)
    fig.add_hline(y=high_thr, line_color="rgba(200,200,200,0.5)", line_dash="dot", line_width=1.5)
    fig.add_hline(y=low_thr, line_color="rgba(200,200,200,0.5)", line_dash="dot", line_width=1.5)

    if not out_of_bounds.empty:
        fig.add_trace(
            go.Scatter(
                x=out_of_bounds[time_col],
                y=out_of_bounds[predicted_col],
                mode='markers',
                marker=dict(color='red', size=6, symbol='circle'),
                name="Predicted out of bounds"
            )
        )

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


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
    """
    Compare the distributions of selected numeric columns in the full dataset
    (df_all) vs. the current window (df_window), using probability density
    rather than absolute counts.

    We use histnorm='probability density' so that the integral of each histogram is 1, rather than counting the raw number of samples.
    """
    if not columns:
        st.write("No columns selected for distribution comparison.")
        return

    n = len(columns)
    rows = math.ceil(n / 4)
    fig = make_subplots(rows=rows, cols=4, subplot_titles=columns)

    for i, col in enumerate(columns, start=1):
        row = math.ceil(i / 4)
        col_pos = i - (row - 1) * 4

        # Plot the overall distribution for the column (density normalized)
        if col in df_all.columns and pd.api.types.is_numeric_dtype(df_all[col]):
            fig.add_trace(
                go.Histogram(
                    x=df_all[col].dropna(),
                    histnorm='probability density',
                    name='Overall',
                    marker_color='royalblue',
                    opacity=0.6,
                    showlegend=(i == 1)
                ),
                row=row,
                col=col_pos
            )

        # Plot the window distribution for the column (density normalized)
        if col in df_window.columns and pd.api.types.is_numeric_dtype(df_window[col]):
            fig.add_trace(
                go.Histogram(
                    x=df_window[col].dropna(),
                    histnorm='probability density',
                    name='Window',
                    marker_color='tomato',
                    opacity=0.6,
                    showlegend=(i == 1)
                ),
                row=row,
                col=col_pos
            )

        # Label the x-axis with the column name
        fig.update_xaxes(title_text=col, row=row, col=col_pos)

    # Overlay the histograms so they can be compared easily
    fig.update_layout(barmode='overlay', height=400 * rows)

    # Set to True so the chart fills the entire horizontal space of the container
    st.plotly_chart(fig, use_container_width=True)

def plot_features_stacked_synced_scale(df: pd.DataFrame, features: list, y_range: tuple[float, float], time_col: str = "timestamp", z_threshold: float = 3.0):
    df_plot = df.copy()
    if isinstance(df_plot[time_col].dtype, pd.DatetimeTZDtype):
        df_plot[time_col] = df_plot[time_col].dt.tz_convert(None)
    n = len(features)
    if n == 0:
        st.write("No key variables to display.")
        return

    color_palette = [
        "royalblue",      # Blue
        "tomato",         # Red-orange
        "mediumseagreen", # Green
        "mediumpurple",   # Purple
        "darkorange",     # Orange
        "steelblue",      # Steel blue
        "indianred",      # Indian red
        "darkturquoise"   # Turquoise
    ]
    
    outlier_colors = [
        "darkblue",
        "darkred",
        "darkgreen",
        "indigo",
        "darkorange",
        "midnightblue",
        "maroon",
        "teal"
    ]

    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=features)
    
    for i, feat in enumerate(features, start=1):
        if feat in df_plot.columns and not df_plot[feat].dropna().empty:
            line_color = color_palette[(i-1) % len(color_palette)]
            outlier_color = outlier_colors[(i-1) % len(outlier_colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=df_plot[time_col], 
                    y=df_plot[feat], 
                    mode="lines", 
                    name=feat, 
                    line=dict(color=line_color, width=2)
                ),
                row=i, col=1
            )
            
            feat_data = df_plot[[time_col, feat]].dropna()
            if not feat_data.empty:
                values = feat_data[feat].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if std_val > 0:
                    z_scores = np.abs((values - mean_val) / std_val)
                    outlier_mask = z_scores > z_threshold
                    outliers = feat_data[outlier_mask]
                    
                    if not outliers.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=outliers[time_col], 
                                y=outliers[feat], 
                                mode="markers",
                                name=f"{feat} outliers",
                                marker=dict(
                                    color="red", 
                                    size=8, 
                                    symbol="x", 
                                    line=dict(width=1.5, color=outlier_color)
                                ),
                                showlegend=False
                            ),
                            row=i, col=1
                        )
                
                feat_min = float(values.min())
                feat_max = float(values.max())
                
                if feat_min == feat_max:
                    delta = 1.0 if feat_min == 0 else abs(feat_min) * 0.1
                    feat_min -= delta
                    feat_max += delta
                else:
                    padding = (feat_max - feat_min) * 0.05
                    feat_min -= padding
                    feat_max += padding
                
                fig.update_yaxes(range=[feat_min, feat_max], row=i, col=1)

    fig.update_layout(
        height=max(180 * n, 420), 
        showlegend=False, 
        margin=dict(l=20, r=20, t=50, b=20), 
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
def plot_feature_importances(model, feature_columns, top_n=None):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_).astype(float))
    else:
        st.write("Model does not provide feature importances.")
        return

    k = min(len(importances), len(feature_columns))
    df_imp = pd.DataFrame({
        "feature": feature_columns[:k],
        "importance": importances[:k]
    }).dropna()

    df_imp = df_imp.sort_values("importance", ascending=False)
    if top_n is not None:
        df_imp = df_imp.head(top_n)

    df_imp = df_imp.sort_values("importance", ascending=True)
    fig = px.bar(df_imp, x="importance", y="feature", orientation="h", title="Feature Importances (RandomForest)")
    fig.update_yaxes(title="Features", categoryorder="array", categoryarray=df_imp["feature"].tolist())
    fig.update_xaxes(title="Importance")
    fig.update_layout(margin=dict(l=160, r=24, t=40, b=24), height=max(40 * len(df_imp) + 120, 320), bargap=0.2)
    st.plotly_chart(fig, use_container_width=True)
    
def plot_anomaly_detection_graph(df: pd.DataFrame, anomaly_cols: list):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(anomaly_cols):
        color = colors[i % len(colors)]
        normal = df[~df[f"{col}_anomaly"]]
        anomalous = df[df[f"{col}_anomaly"]]
        fig.add_trace(
            go.Scatter(
                x=normal['timestamp'], 
                y=normal[col],
                mode='lines',
                name=f"{col} Normal",
                line=dict(color=color)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=anomalous['timestamp'], 
                y=anomalous[col],
                mode='markers',
                name=f"{col} Anomaly",
                marker=dict(color=color, size=10, symbol='x', line=dict(width=1, color='black'))
            )
        )
    fig.update_layout(title="Anomaly Detection Over Time", height=300)
    st.plotly_chart(fig, use_container_width=True)

def plot_features_2x4_subplots_anomaly(df: pd.DataFrame, features: list):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    fig = make_subplots(rows=2, cols=4, subplot_titles=features)
    row_col_map = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4)]
    for i, feat in enumerate(features):
        row, col = row_col_map[i]
        anom_col = f"{feat}_anomaly"
        mask_anom = df.get(anom_col, False)
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
    st.plotly_chart(fig, use_container_width=True)

def detect_anomalies_for_features(df: pd.DataFrame, features: list, z_threshold=3.0) -> pd.DataFrame:
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

def compute_performance_metrics(df: pd.DataFrame, target_col: str = "moisture_in_z0", pred_col: str = "predicted_moisture") -> dict:
    df_valid = df.dropna(subset=[target_col, pred_col])
    if df_valid.empty:
        return {"mse": None, "mae": None, "mape": None, "r2": None}
    y_true = df_valid[target_col]
    y_pred = df_valid[pred_col]
    mse_val = mean_squared_error(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    r2_val  = r2_score(y_true, y_pred)
    mape_val = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    return {"mse": mse_val, "mae": mae_val, "mape": mape_val, "r2": r2_val}

def get_top_n_features(model, feature_names, n=8):
    if not hasattr(model, 'feature_importances_'):
        return []
    importances = model.feature_importances_
    indices_desc = np.argsort(importances)[::-1]
    top_indices = indices_desc[:n]
    return [feature_names[i] for i in top_indices]

def apply_feature_adjustments_preview(data_window: pd.DataFrame) -> pd.DataFrame:
    adjusted_df = data_window.copy()
    feature_adjustments = st.session_state.get("feature_adjustments", {})
    if not feature_adjustments:
        return adjusted_df
    if isinstance(adjusted_df["timestamp"].dtype, pd.DatetimeTZDtype):
        adjusted_df["timestamp"] = adjusted_df["timestamp"].dt.tz_convert(None)
    for feat, adj in feature_adjustments.items():
        if feat in adjusted_df.columns and adj != 0:
            adjusted_df[feat] = adjusted_df[feat] * (1 + adj / 100.0)
    return adjusted_df

def apply_future_feature_adjustments(data_window: pd.DataFrame) -> pd.DataFrame:
    adjusted_df = data_window.copy()

    feature_adjustments = st.session_state.get("feature_adjustments", {})
    if not any(adj != 0 for adj in feature_adjustments.values()):
        return adjusted_df

    adj_start = st.session_state.get("adjustment_start_time")
    if adj_start is None:
        adj_start = adjusted_df["timestamp"].max()

    if isinstance(adj_start, pd.Timestamp) and adj_start.tzinfo is not None:
        adj_start = adj_start.tz_convert(None)

    if isinstance(adjusted_df["timestamp"].dtype, pd.DatetimeTZDtype):
        adjusted_df["timestamp"] = adjusted_df["timestamp"].dt.tz_convert(None)

    for feat, adj in feature_adjustments.items():
        if feat in adjusted_df.columns and adj != 0:
            mask = adjusted_df["timestamp"] > adj_start
            adjusted_df.loc[mask, feat] = adjusted_df.loc[mask, feat] * (1 + adj / 100.0)

    return adjusted_df


###############################################################################
# 5) Data Exploration Tab
###############################################################################
@fragment
def data_exploration_tab(df_all: pd.DataFrame, model, feature_columns):
    if "top_features" not in st.session_state or not st.session_state.top_features:
        if model is not None and "feature_columns" in st.session_state:
            st.session_state.top_features = get_top_n_features(model, st.session_state.feature_columns, n=8)

    window_duration = st.session_state.window_duration
    current_offset = st.session_state.get("current_offset", timedelta(0))
    data_window = get_sliding_window_data(df_all, current_offset, window_duration)
    default_top_features = st.session_state.get("top_features", None)

    def _mark_explore_active():
        st.session_state["active_tab_name"] = "Data Exploration"

    analysis_options = [
        "Correlation Heatmap (Full Dataset)",
        "Window Distribution Comparison",
        "Anomaly Detection (Window)",
        "Feature Importances",
        "SHAP Summary (beeswarm)",
        "Dataframe & Anomalies (Last 5 Rows)"
    ]
    selected_analyses = st.multiselect(
        "Select Analyses to Display",
        analysis_options,
        default=[
            "Window Distribution Comparison",
            "Anomaly Detection (Window)",
            "Feature Importances",
            "SHAP Summary (beeswarm)",
            "Correlation Heatmap (Full Dataset)",
            "Dataframe & Anomalies (Last 5 Rows)"
        ],
        key="explore_analyses",
        on_change=_mark_explore_active
    )

    correlation_selected = "Correlation Heatmap (Full Dataset)" in selected_analyses
    feature_importances_selected = "Feature Importances" in selected_analyses

    if correlation_selected and feature_importances_selected:
        left_col, right_col = st.columns(2)
        with left_col:
            numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
            default_corr = default_top_features if default_top_features and len(default_top_features) >= 2 else (numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
            chosen_corr_cols = st.multiselect("Columns for correlation heatmap", numeric_cols, default=default_corr, key="corr_cols_expl")
            st.write("#### Correlation Heatmap")
            plot_correlation_custom(df_all, chosen_corr_cols)
        with right_col:
            st.write("#### Feature Importances (Model)")
            plot_feature_importances(model, feature_columns, top_n=10)
    elif correlation_selected:
        numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        default_corr = default_top_features if default_top_features and len(default_top_features) >= 2 else (numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
        chosen_corr_cols = st.multiselect("Columns for correlation heatmap", numeric_cols, default=default_corr, key="corr_cols_only")
        st.write("#### Correlation Heatmap")
        plot_correlation_custom(df_all, chosen_corr_cols)
    elif feature_importances_selected:
        st.write("#### Feature Importances (Model)")
        plot_feature_importances(model, feature_columns, top_n=10)

    if "Window Distribution Comparison" in selected_analyses:
        st.write("### Window Distribution Comparison")
        default_dist_cols = default_top_features if default_top_features and len(default_top_features) >= 2 else ['raw_in_left', 'raw_in_right', 'raw_out_left', 'raw_out_right']
        selected_cols = st.multiselect("Select columns for distribution plots", df_all.select_dtypes(include=[np.number]).columns.tolist(), default=default_dist_cols, key="dist_cols")
        plot_distribution_comparison_2x4(df_all, data_window, selected_cols)

    if "Anomaly Detection (Window)" in selected_analyses:
        st.write("### Anomaly Detection (Window)")
        rf_model = st.session_state.get("model", None)
        window_with_preds = predict(rf_model, data_window)
        top_features = st.session_state.get("top_features", [])
        anomalies_df_window = detect_anomalies_for_features(window_with_preds, top_features, z_threshold=3.0)
        st.dataframe(anomalies_df_window.tail(200), use_container_width=True)

    if "Dataframe & Anomalies (Last 5 Rows)" in selected_analyses:
        st.write("### Dataframe & Anomalies (Last 5 Rows)")
        rf_model = st.session_state.get("model", None)
        window_with_preds = predict(rf_model, data_window)
        top_features = st.session_state.get("top_features", [])
        anomalies_df = detect_anomalies_for_features(window_with_preds, top_features, z_threshold=3.0)
        def highlight_anomaly_row(row):
            return ["background-color: yellow" if row.get("any_anomaly", False) else "" for _ in row]
        df_last5 = anomalies_df.sort_values('timestamp').tail(5)
        cols = list(df_last5.columns)
        desired_order = []
        if 'moisture_in_z0' in cols:
            desired_order.append('moisture_in_z0')
        if 'predicted_moisture' in cols:
            desired_order.append('predicted_moisture')
        if 'timestamp' in cols:
            desired_order.append('timestamp')
        for col in cols:
            if col not in desired_order:
                desired_order.append(col)
        df_last5 = df_last5[desired_order]
        st.write(df_last5.style.apply(highlight_anomaly_row, axis=1))

    if "SHAP Summary (beeswarm)" in selected_analyses:
        if model is not None and feature_columns:
            with st.spinner("Computing SHAP..."):
                X = df_all[feature_columns].select_dtypes(include=[np.number]).dropna()
                if len(X) > 0:
                    n_rows = min(len(X), 1200)
                    X_sample = X.sample(n_rows, random_state=7)
                    bg_rows = min(len(X_sample), 240)
                    bg = X_sample.sample(bg_rows, random_state=11)
                    explainer = shap.Explainer(model, bg)
                    shap_values = explainer(X_sample, check_additivity=False)
                    fig_bee = plt.figure()
                    fig_bee.set_size_inches(6.6, 3.8)
                    shap.summary_plot(shap_values.values, X_sample, max_display=20, show=False)
                    st.pyplot(fig_bee, use_container_width=True)
                else:
                    st.write("No numeric features available for SHAP.")

###############################################################################
# 6) Predictions Tabs (Streaming and Paused)
###############################################################################
@fragment(run_every=3)
def predictions_tab_streaming(df_all: pd.DataFrame):
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True).dt.tz_convert(None)
    window_duration = st.session_state.window_duration
    if "current_offset" not in st.session_state:
        st.session_state.current_offset = timedelta(0)
    data_window = get_sliding_window_data(df_all, st.session_state.current_offset, window_duration)

    rf_model = st.session_state.get("model", None)

    baseline_preds = predict(rf_model, data_window)
    top_features = st.session_state.get("top_features", [])

    df_ana = detect_anomalies_for_features(baseline_preds, top_features, z_threshold=3.0)
    st.session_state["df_main_anomalies"] = df_ana

    pred_series = df_ana.get("predicted_moisture", pd.Series(dtype=float)).dropna()
    if pred_series.empty:
        y_range = (0.0, 1.0)
    else:
        ymin, ymax = float(pred_series.min()), float(pred_series.max())
        if ymin == ymax:
            ymin, ymax = ymin - 1.0, ymax + 1.0
        y_range = (ymin, ymax)

    adjustments_nonzero = any(v != 0 for v in st.session_state.get("feature_adjustments", {}).values())
    overlay_df = None
    if adjustments_nonzero:
        preview_adjusted = apply_feature_adjustments_preview(data_window)
        overlay_df = predict(rf_model, preview_adjusted)

    st.markdown("### Predicted Moisture")
    plot_timeseries_with_prediction_interactive(
        df=df_ana,
        model=rf_model,
        feature_columns=st.session_state.get("feature_columns", []),
        include_actual=False,
        overlay_df=overlay_df
    )

    with st.expander("Key Variables (Top 8)"):
        if not top_features:
            st.warning("Top features are not computed yet. Start streaming to compute them.")
        else:
            plot_features_stacked_synced_scale(df_ana, top_features, y_range)

    st.session_state.current_offset += timedelta(seconds=10)


@fragment(run_every=None)
def predictions_tab_paused(df_all: pd.DataFrame):
    df_all = df_all.copy()
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True).dt.tz_convert(None)

    window_duration = st.session_state.window_duration
    current_offset = st.session_state.get("current_offset", timedelta(0))
    data_window = get_sliding_window_data(df_all, current_offset, window_duration)

    rf_model = st.session_state.get("model", None)

    baseline_preds = predict(rf_model, data_window)
    top_features = st.session_state.get("top_features", [])

    df_ana = detect_anomalies_for_features(baseline_preds, top_features, z_threshold=3.0)

    pred_series = df_ana.get("predicted_moisture", pd.Series(dtype=float)).dropna()
    if pred_series.empty:
        y_range = (0.0, 1.0)
    else:
        ymin, ymax = float(pred_series.min()), float(pred_series.max())
        if ymin == ymax:
            ymin, ymax = ymin - 1.0, ymax + 1.0
        y_range = (ymin, ymax)

    adjustments_nonzero = any(v != 0 for v in st.session_state.get("feature_adjustments", {}).values())
    overlay_df = None
    if adjustments_nonzero:
        preview_adjusted = apply_feature_adjustments_preview(data_window)
        overlay_df = predict(rf_model, preview_adjusted)

    st.markdown("### Predicted Moisture")
    plot_timeseries_with_prediction_interactive(
        df=df_ana,
        model=rf_model,
        feature_columns=st.session_state.get("feature_columns", []),
        include_actual=False,
        overlay_df=overlay_df
    )

    with st.expander("Key Variables (Top 8)"):
        if not top_features:
            st.warning("Top features are not computed yet. Start streaming to compute them.")
        else:
            plot_features_stacked_synced_scale(df_ana, top_features, y_range)

###############################################################################
# 7) Model Performance Tab
###############################################################################
def model_performance_tab(df_all: pd.DataFrame, model, feature_columns):
    if model is None:
        st.warning("No trained model available. Please train the model first.")
        return
    df_all_preds = predict(model, df_all)
    df_all_preds.rename(columns={"predicted_moisture": "predicted_moisture_full"}, inplace=True)
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
    
    with col1:
        st.markdown("<h1 style='color:white;'>Full Dataset Metrics</h1>", unsafe_allow_html=True)
        if full_metrics["mse"] is None:
            st.write("No valid rows for full-dataset metric computation.")
        else:
            st.markdown(f"<h2 style='color:tomato;'>MAE (Full): {full_metrics['mae']:.4f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:tomato;'>MAPE (Full): {full_metrics['mape']:.2f}%</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:royalblue;'>MSE (Full): {full_metrics['mse']:.4f}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:royalblue;'>R² (Full): {full_metrics['r2']:.4f}</h3>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h1 style='color:white;'>Sliding Window Metrics</h1>", unsafe_allow_html=True)
        if window_metrics["mse"] is None:
            st.write("No valid rows in the sliding window for metric computation.")
        else:
            st.markdown(f"<h2 style='color:tomato;'>MAE (Window): {window_metrics['mae']:.4f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:tomato;'>MAPE (Window): {window_metrics['mape']:.2f}%</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:royalblue;'>MSE (Window): {window_metrics['mse']:.4f}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:royalblue;'>R² (Window): {window_metrics['r2']:.4f}</h3>", unsafe_allow_html=True)
            
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


###############################################################################
# 8) Predictions Tab Controller
###############################################################################
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
    if "feature_adjustments" not in st.session_state:
        st.session_state.feature_adjustments = {}
    if "btn_version" not in st.session_state:
        st.session_state.btn_version = 0

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
            # Convert timestamp column to naive datetime for filtering
            df_all_naive = df_all.copy()
            if isinstance(df_all_naive['timestamp'].dtype, pd.DatetimeTZDtype):
                df_all_naive['timestamp'] = df_all_naive['timestamp'].dt.tz_convert(None)
            retrain_data = df_all_naive[
                (df_all_naive['timestamp'] >= fixed_start_datetime) &
                (df_all_naive['timestamp'] <= fixed_end_datetime)
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
        
    with st.expander("Influence Top Feature Values", expanded=True):
        cols = st.columns(4)
        adjustments = {}
        for i, feat in enumerate(st.session_state.top_features):
            key = f"adjust_{feat}"
            val = cols[i % 4].slider(
                feat, -20, 20,
                value=st.session_state.feature_adjustments.get(feat, 0),
                step=5, key=key
            )
            adjustments[feat] = val
        st.session_state.feature_adjustments = adjustments

        apply_key = f"apply_{st.session_state.btn_version}"
        if st.button("Apply", key=apply_key):
            st.session_state.adjustment_start_time = (
                st.session_state.df_main_anomalies["timestamp"].max()
            )
            st.session_state.btn_version += 1
            st.rerun()

        reset_key = f"reset_{st.session_state.btn_version}"
        if st.button("Reset", key=reset_key):
            for feat in st.session_state.top_features:
                st.session_state.pop(f"adjust_{feat}", None)
            st.session_state.feature_adjustments = {
                feat: 0 for feat in st.session_state.top_features
            }
            st.session_state.pop("adjustment_start_time", None)
            st.session_state.btn_version += 1
            st.rerun()

###############################################################################
# 9) Main App
###############################################################################
def main():
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

    # Center the logo using column layout
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <a href="http://www.miningfulstudio.eu/">
                <img src="app/static/Miningful_NoBG_WhiteText.png" 
                     alt="Miningful" 
                     style="width:120px;" />
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### Predictive Maintenance Demo")

    feature_columns = [
        'shrink_raw_in_left', 'shrink_raw_in_right',
        'shrink_raw_out_left', 'shrink_raw_out_right',
        'paperwidth_in', 'paperwidth_out',

        'temperature_in_z0', 'temperature_out_z0',
        'temperature_in_z1', 'temperature_out_z1',
        'temperature_in_z2', 'temperature_out_z2',
        'temperature_in_z3', 'temperature_out_z3',
        'temperature_in_z4', 'temperature_out_z4',
        'temperature_in_z5', 'temperature_out_z5',
        'temperature_in_z6', 'temperature_out_z6',
        'temperature_in_z7', 'temperature_out_z7',
        'temperature_in_z8', 'temperature_out_z8',
        'temperature_in_z9', 'temperature_out_z9'
    ]
    st.session_state.feature_columns = feature_columns

    if "stream_data" not in st.session_state:
        st.session_state.stream_data = load_data_remote()
    df_all = st.session_state.stream_data

    if "model" not in st.session_state:
        with st.spinner("Training Random Forest model..."):
            model, mse = train_model(df_all)
            st.session_state.model = model
            st.session_state.model_mse = mse

    if st.session_state.model is not None:
        st.markdown("<span style='color:green;font-weight:bold;'>Model training complete!</span>", unsafe_allow_html=True)
    else:
        st.warning("Not enough data to train the model.")

    if st.session_state.model is not None:
        st.session_state.top_features = get_top_n_features(st.session_state.model, feature_columns, n=8)

    if "df_main_anomalies" not in st.session_state:
        max_ts = df_all["timestamp"].max()
        default_window = get_sliding_window_data(df_all, timedelta(0), 1.0)
        st.session_state["df_main_anomalies"] = predict(st.session_state.model, default_window)

    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "current_offset" not in st.session_state:
        st.session_state.current_offset = timedelta(0)

    # Updated tab names
    tab1, tab2, tab3 = st.tabs(["Monitor", "Predictions", "Data Exploration"])
    with tab1:
        predictions_tab_controller(df_all)  # 'Monitor' tab
    with tab2:
        model_performance_tab(df_all, st.session_state.model, feature_columns)  # 'Predictions' tab
    with tab3:
        data_exploration_tab(df_all, st.session_state.model, feature_columns)   # 'Data Exploration' tab

if __name__ == "__main__":
    main()
