import streamlit as st
from streamlit.runtime.fragment import fragment
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

def find_index_for_time(df: pd.DataFrame, t: pd.Timestamp) -> int:
    matching = df.index[df['timestamp'] >= t]
    return matching[0] if len(matching) > 0 else len(df) - 1

def get_window_data(df, end_index: int, days: int):
    """For Predictions tab: slice last 'days' from end_index."""
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
    """Plot actual vs. predicted moisture as lines in Plotly."""
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
# 4) Data Exploration & Visualization
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
    Compare distribution of some columns in the entire dataset vs the window subset.
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
# 5) Anomaly Detection: Z-Score on the Top 8 Features
###############################################################################
def detect_anomalies_for_features(df: pd.DataFrame, features: list, z_threshold=3.0) -> pd.DataFrame:
    """
    Returns a copy of df with:
      - <feature>_anomaly for each of the given `features`
      - any_anomaly = True if any of those features is anomalous in that row
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
# 6) Feature Importances
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

def get_top_n_features(model, feature_names, n=8):
    """Return top-n features based on feature_importances_, descending."""
    if not hasattr(model, 'feature_importances_'):
        return []
    importances = model.feature_importances_
    indices_desc = np.argsort(importances)[::-1]
    top_indices = indices_desc[:n]
    return [feature_names[i] for i in top_indices]

###############################################################################
# 7) Key Variables Plot: 2×4 subplots, sliding last 50 rows, weekend skip
###############################################################################
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_features_2x4_subplots_anomaly(df: pd.DataFrame, features: list):
    """
    Creates a 2×4 Plotly subplot layout for the given features:
      - Show a continuous line for all data in the slice (normal+anomaly),
      - Overlay anomaly points in red "X",
      - Remove weekends from the x-axis by enumerating each hour in Sat/Sun.
    """
    # Convert to timezone-naive
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)

    # Build weekend skip list
    start_date = df["timestamp"].min().floor('D')
    end_date = df["timestamp"].max().ceil('D')
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')
    weekend_days = all_days[all_days.dayofweek.isin([5,6])]  # 5=Sat,6=Sun

    to_skip = []
    for d in weekend_days:
        day_hours = pd.date_range(d, d + pd.Timedelta(hours=23), freq='H')
        to_skip.extend(day_hours)

    fig = make_subplots(rows=2, cols=4, subplot_titles=features)
    row_col_map = [
        (1,1), (1,2), (1,3), (1,4),
        (2,1), (2,2), (2,3), (2,4)
    ]

    for i, feat in enumerate(features):
        row, col = row_col_map[i]
        anom_col = f"{feat}_anomaly"
        mask_anom = df.get(anom_col, False)

        # A) One continuous line for the entire slice
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[feat],
                mode="lines",
                name=f"{feat} (all)"
            ),
            row=row, col=col
        )

        # B) Anomaly points only
        df_anom = df[mask_anom]
        fig.add_trace(
            go.Scatter(
                x=df_anom["timestamp"],
                y=df_anom[feat],
                mode="markers",
                name=f"{feat} (anomaly)",
                marker=dict(color="red", size=6, symbol="x")
            ),
            row=row, col=col
        )

    # Weekend skipping
    for axis_name in fig.layout:
        if axis_name.startswith("xaxis"):
            fig.layout[axis_name].rangebreaks = [dict(values=to_skip)]
            fig.layout[axis_name].type = "date"

    fig.update_layout(
        autosize=False,
        width=1200,
        height=400,
        showlegend=False,
        title_text="Key Variables (sliding last 50 rows, skip weekends, anomalies overlaid)",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=False)

###############################################################################
# 8) Data Exploration Fragment
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

    # 1) Distributions
    if "Distributions (Full Dataset)" in selected_analyses:
        numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        chosen_dist_cols = st.multiselect(
            "Choose columns for distribution plots (Full Dataset)",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        st.write("### Distributions (Full Dataset)")
        plot_distributions_custom(df_all, chosen_dist_cols)

    # 2) Correlation
    if "Correlation Heatmap (Full Dataset)" in selected_analyses:
        numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        chosen_corr_cols = st.multiselect(
            "Choose columns for correlation heatmap",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
        )
        st.write("### Correlation Heatmap")
        plot_correlation_custom(df_all, chosen_corr_cols)

    # 3) Window Dist. Compare
    if "Window Distribution Comparison" in selected_analyses:
        st.write("### Window Distribution Comparison")
        plot_distribution_comparison(df_all, data_window)

    # 4) Anomaly Detection
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
            anomalies_df = detect_anomalies_for_features(data_window, selected_anomaly_cols, z_threshold=z_threshold)
            anomaly_rate = anomalies_df['any_anomaly'].mean()
            st.write(f"**Anomaly Rate (Window)**: {anomaly_rate*100:.2f}%")

            anomaly_rows = anomalies_df[anomalies_df['any_anomaly'] == True]
            if not anomaly_rows.empty:
                st.write("**Anomalous rows (up to 20 displayed)**:")
                st.dataframe(anomaly_rows.head(20))
            else:
                st.write("No anomalies found with current threshold and columns.")

            # (Optional) Plot single-col anomalies

    # 5) Feature Importances
    if "Feature Importances" in selected_analyses and model is not None:
        st.write("### Feature Importances (Model)")
        plot_feature_importances(model, feature_columns)

###############################################################################
# 9) Predictions Tab (Streaming & Paused)
###############################################################################
@fragment(run_every=10)
def predictions_tab_streaming(df_all: pd.DataFrame):
    """Re-runs every 10 seconds, shows last 5 rows with anomaly highlighting."""
    if st.session_state.current_index >= len(df_all):
        st.warning("Reached the end of the dataset.")
        st.session_state.streaming = False
        return

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True).dt.tz_convert(None)
    data_window_size = st.session_state.get("demo_window_size", 3)
    data_window = get_window_data(df_all, st.session_state.current_index, data_window_size)
    model = st.session_state.get("model", None)
    data_with_preds = predict(model, data_window)

    # Detect anomalies for top 8 only
    top_features = st.session_state.get("top_features", [])
    df_ana = detect_anomalies_for_features(data_with_preds, top_features, z_threshold=3.0)
    st.session_state["df_main_anomalies"] = df_ana

    # Highlight row if any_anomaly == True
    def highlight_anomaly_row(row):
        if row.get("any_anomaly", False):
            return ["background-color: yellow"] * len(row)
        else:
            return ["" for _ in row]

    last_5 = df_ana.tail(5)
    df_styled = last_5.style.apply(highlight_anomaly_row, axis=1)
    

    col_left, col_right = st.columns([1,2])
    with col_left:
        st.markdown("#### Predictions - Streaming (every 10s)")
        st.write(df_styled)

        # Show which top-8 columns are anomalous for these rows
        anomaly_rows = last_5[last_5["any_anomaly"] == True]
        if not anomaly_rows.empty:
            st.write("**Anomaly columns (within top 8)** for these rows:")
            for idx, row in anomaly_rows.iterrows():
                anom_cols = [f for f in top_features if row.get(f"{f}_anomaly", False)]
                if anom_cols:
                    st.write(f"Row {idx}: {', '.join(anom_cols)}")
        else:
            st.write("No anomalies in last 5 rows (top 8 features).")

    with col_right:
        plot_timeseries_with_prediction_interactive(df_ana)

    # Advance index by 1
    st.session_state.current_index += 1

@fragment(run_every=None)
def predictions_tab_paused(df_all: pd.DataFrame):
    """Paused state: never auto re-runs."""
    end_idx = min(st.session_state.current_index, len(df_all) - 1)
    model = st.session_state.get("model", None)
    data_window_size = st.session_state.get("demo_window_size", 3)
    data_window = get_window_data(df_all, end_idx, data_window_size)
    data_with_preds = predict(model, data_window)

    top_features = st.session_state.get("top_features", [])
    df_ana = detect_anomalies_for_features(data_with_preds, top_features, z_threshold=3.0)

    def highlight_anomaly_row(row):
        if row.get("any_anomaly", False):
            return ["background-color: yellow"] * len(row)
        else:
            return ["" for _ in row]

    last_5 = df_ana.tail(5)
    df_styled = last_5.style.apply(highlight_anomaly_row, axis=1)

    col_left, col_right = st.columns([1,2])
    with col_left:
        st.markdown("#### Predictions - Paused")
        if not df_ana.empty:
            st.write(df_styled)

            # Show which top-8 columns are anomalous
            anomaly_rows = last_5[last_5["any_anomaly"] == True]
            if not anomaly_rows.empty:
                st.write("**Anomaly columns (within top 8)**:")
                for idx, row in anomaly_rows.iterrows():
                    anom_cols = [f for f in top_features if row.get(f"{f}_anomaly", False)]
                    if anom_cols:
                        st.write(f"Row {idx}: {', '.join(anom_cols)}")
            else:
                st.write("No anomalies in last 5 rows (top 8 features).")
        else:
            st.write("No data available in the current window.")

    with col_right:
        plot_timeseries_with_prediction_interactive(df_ana)

###############################################################################
# 10) Key Variables Tab (Sliding Last 50 Rows)
###############################################################################
WINDOW_SIZE_KEYVARS = 5
@fragment(run_every=None)
def key_variables_tab_paused(df_all: pd.DataFrame):
    """Pauses at the current_index, shows a 50-row window using shared anomaly DataFrame."""
    st.write("### Key Variables - Paused (Using Shared Anomalies)")

    if "df_main_anomalies" not in st.session_state:
        st.warning("No main anomaly data in session_state. Please run Predictions tab first.")
        return
    
    df_main_ana = st.session_state["df_main_anomalies"]
    top_features = st.session_state.get("top_features", [])

    if "current_index" not in st.session_state:
        st.session_state.current_index = 49  # or any suitable default

    current_idx = st.session_state.current_index
    end_idx = min(current_idx, len(df_main_ana) - 1)
    start_idx = max(0, end_idx - (WINDOW_SIZE_KEYVARS - 1))

    df_window = df_main_ana.iloc[start_idx : end_idx + 1].copy()
    if df_window.empty:
        st.warning("No data in this 50-row window.")
        return

    # Reuse your existing subplot function that expects anomaly columns
    plot_features_2x4_subplots_anomaly(df_window, top_features)

###############################################################################
# 11) Controllers
###############################################################################
def predictions_tab_controller(df_all: pd.DataFrame):
    st.subheader("Predictions and Real-Time Stream")

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.streaming:
            if st.button("▶️ Start Streaming"):
                # Compute top 8 if not done
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

###############################
# KEY VARIABLES TAB CONTROLLER
###############################
def key_variables_tab_controller(df_all: pd.DataFrame):
    st.subheader("Key Variables (Top 8 Features)")
    if st.session_state.streaming:
        st.write("**Streaming is ON.**")
        key_variables_tab_streaming(df_all) 
    else:
        st.write("**Streaming is paused.**")
        key_variables_tab_paused(df_all)
        

@fragment(run_every=10)
def key_variables_tab_streaming(df_all: pd.DataFrame):
    st.write("### Key Variables - Streaming (Using Shared Anomalies)")

    if "current_index" not in st.session_state:
        st.session_state.current_index = 49

    # 1) Grab the main anomaly-labeled df from session_state
    if "df_main_anomalies" not in st.session_state:
        st.warning("No anomaly data found in session_state. Please run Predictions tab first.")
        return

    df_main_ana = st.session_state["df_main_anomalies"]
    top_features = st.session_state.get("top_features", [])

    # 2) Slide your window in the *already-analyzed* dataframe
    current_idx = st.session_state.current_index
    end_idx = min(current_idx, len(df_main_ana) - 1)
    start_idx = max(0, end_idx - (WINDOW_SIZE_KEYVARS - 1))

    df_window = df_main_ana.iloc[start_idx : end_idx + 1].copy()
    if df_window.empty:
        st.warning("No data in this 50-row window.")
        return

    # 3) Simply plot these anomalies (no new calls to detect_anomalies_for_features)
    plot_features_2x4_subplots_anomaly(df_window, top_features)

    # 4) Advance index
    if current_idx < len(df_main_ana) - 1:
        st.session_state.current_index += 1
    else:
        st.warning("Reached end of dataset.")


def plot_features_2x4_no_weekends(df: pd.DataFrame, features: list):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Convert timestamp to naive (no timezone)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)

    fig = make_subplots(rows=2, cols=4, subplot_titles=features)
    layout_map = [(1,1),(1,2),(1,3),(1,4),
                  (2,1),(2,2),(2,3),(2,4)]

    for i, feat in enumerate(features):
        row, col = layout_map[i]
        anom_col = f"{feat}_anomaly"
        mask_anom = df.get(anom_col, False)

        # Main line
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[feat],
                mode="lines",
                name=feat
            ),
            row=row, col=col
        )

        # Red "X" anomaly points
        df_anom = df[mask_anom]
        fig.add_trace(
            go.Scatter(
                x=df_anom["timestamp"],
                y=df_anom[feat],
                mode="markers",
                marker=dict(color="red", size=7, symbol="x"),
                name=f"{feat}_anomaly"
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Key Variables"
    )
    st.plotly_chart(fig, use_container_width=True)
    

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_performance_metrics(df: pd.DataFrame, 
                                target_col: str = "moisture_in_z0", 
                                pred_col: str = "predicted_moisture") -> dict:
    """
    Compute MSE, MAE, and R-squared for rows that have both target_col and pred_col (non-null).
    Return a dict with {mse, mae, r2}.
    """
    df_valid = df.dropna(subset=[target_col, pred_col])
    if df_valid.empty:
        return {"mse": None, "mae": None, "r2": None}
    
    y_true = df_valid[target_col]
    y_pred = df_valid[pred_col]
    
    mse_val = mean_squared_error(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    r2_val  = r2_score(y_true, y_pred)
    return {"mse": mse_val, "mae": mae_val, "r2": r2_val}

@fragment(run_every=None)
def model_performance_tab(df_all: pd.DataFrame, model, feature_columns):
    st.subheader("Model Performance")

    if model is None:
        st.warning("No trained model available. Please train the model first.")
        return
    
    # 1) Full-dataset prediction
    # ----------------------------------
    df_all_preds = predict(model, df_all)  # uses your existing function
    # We'll rename the column to avoid overwriting anything
    df_all_preds.rename(
        columns={"predicted_moisture": "predicted_moisture_full"}, 
        inplace=True
    )

    full_metrics = compute_performance_metrics(
        df_all_preds, 
        target_col="moisture_in_z0", 
        pred_col="predicted_moisture_full"
    )

    # 2) Current window predictions
    # ----------------------------------
    end_idx = min(st.session_state.current_index, len(df_all) - 1)
    data_window = get_window_data(df_all, end_idx, st.session_state.get("demo_window_size", 3))
    df_window_preds = predict(model, data_window)  # predicted_moisture column
    window_metrics = compute_performance_metrics(
        df_window_preds, 
        target_col="moisture_in_z0", 
        pred_col="predicted_moisture"
    )

    # 3) Display metrics
    # ----------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Full Dataset Metrics")
        if full_metrics["mse"] is None:
            st.write("No valid rows for full-dataset metric computation.")
        else:
            st.metric(label="MSE (Full)", value=f"{full_metrics['mse']:.4f}")
            st.metric(label="MAE (Full)", value=f"{full_metrics['mae']:.4f}")
            st.metric(label="R² (Full)", value=f"{full_metrics['r2']:.4f}")

    with col2:
        st.markdown("### Current Window Metrics")
        if window_metrics["mse"] is None:
            st.write("No valid rows in the current window for metric computation.")
        else:
            st.metric(label="MSE (Window)", value=f"{window_metrics['mse']:.4f}")
            st.metric(label="MAE (Window)", value=f"{window_metrics['mae']:.4f}")
            st.metric(label="R² (Window)", value=f"{window_metrics['r2']:.4f}")

    # 4) Optional: Plot actual vs. predicted lines
    # ----------------------------------
    st.markdown("### Full Dataset: Actual vs. Predicted")
    if "predicted_moisture_full" not in df_all_preds or df_all_preds["predicted_moisture_full"].dropna().empty:
        st.write("No valid predictions for full dataset.")
    else:
        fig_full = px.line(
            df_all_preds.dropna(subset=["predicted_moisture_full", "moisture_in_z0"]),
            x="timestamp",
            y=["moisture_in_z0", "predicted_moisture_full"],
            labels={"value": "Moisture", "variable": "Series", "timestamp": "Time"},
            title="Full Dataset: Actual vs. Predicted Moisture"
        )
        fig_full.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_full, use_container_width=True)

    st.markdown("### Current Window: Actual vs. Predicted")
    if df_window_preds["predicted_moisture"].dropna().empty:
        st.write("No valid predictions for the current window.")
    else:
        fig_window = px.line(
            df_window_preds.dropna(subset=["predicted_moisture", "moisture_in_z0"]),
            x="timestamp",
            y=["moisture_in_z0", "predicted_moisture"],
            labels={"value": "Moisture", "variable": "Series", "timestamp": "Time"},
            title="Current Window: Actual vs. Predicted Moisture"
        )
        fig_window.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_window, use_container_width=True)



###############################################################################
# 12) Main App
###############################################################################
def main():
    st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")

    # Minimize extra spacing with custom CSS
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

    # Initialize current index if needed
    if "initial_window_set" not in st.session_state:
        dataset_start = df_all['timestamp'].min()
        initial_end_time = dataset_start + timedelta(days=demo_window_size)
        idx = find_index_for_time(df_all, initial_end_time)
        st.session_state.current_index = idx
        st.session_state.initial_window_set = True

    # Streaming flag
    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    # Feature columns for top-features logic
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Predictions", 
        "Data Exploration", 
        "Key Variables",
        "Model Performance"
    ])

    with tab1:
        predictions_tab_controller(df_all)
    with tab2:
        data_exploration_tab(df_all, st.session_state.model, feature_columns)
    with tab3:
        key_variables_tab_controller(df_all)
    with tab4:
        model_performance_tab(df_all, st.session_state.model, feature_columns)

if __name__ == "__main__":
    main()
