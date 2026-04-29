
import io
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Drone Trajectory Analysis & Prediction", layout="wide")


DEFAULT_NEW_NAMES = [
    'Serial_No.', 'Timestamp', 'Latitude', 'Longitude', 'Altitude', 'Speed',
    'Heading', 'Pitch', 'Roll', 'Yaw', 'Acceleration_X', 'Acceleration_Y',
    'Acceleration_Z', 'Gyro_X_rad/s', 'Gyro_Y_rad/s', 'Gyro_Z_rad/s',
    'Barometric_Altitude', 'Voltage_V', 'Current_A', 'Satellites', 'GPS_HDOP',
    'Signal_Strength', 'Mode', 'Status', 'Wind_Speed', 'Wind_Direction',
    'Temperature'
]

DEFAULT_OUTLIER_COLUMNS = [
    'Gyro_Z_rad/s', 'Latitude', 'Roll', 'Acceleration_Y',
    'Acceleration_Z', 'Gyro_X_rad/s'
]

NUMERIC_DISTRIBUTION_COLUMNS = [
    'Latitude', 'Longitude', 'Altitude', 'Speed', 'Heading', 'Pitch', 'Roll',
    'Yaw', 'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z',
    'Gyro_X_rad/s', 'Gyro_Y_rad/s', 'Gyro_Z_rad/s', 'Barometric_Altitude',
    'Voltage_V', 'Current_A', 'Satellites', 'GPS_HDOP', 'Signal_Strength',
    'Wind_Speed', 'Wind_Direction', 'Temperature'
]


@dataclass
class ModelArtifacts:
    feature_columns: List[str]
    target_columns: List[str]
    model: LinearRegression
    metrics: Dict[str, float]
    result_df: pd.DataFrame


def rename_columns_safely(df: pd.DataFrame, new_names: List[str]) -> pd.DataFrame:
    if len(new_names) != len(df.columns):
        raise ValueError(
            f"New names count ({len(new_names)}) doesn't match columns ({len(df.columns)})"
        )
    df = df.copy()
    df.columns = new_names
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    report = {}
    for col in df.columns:
        percentage = (df[col].isna().sum() / len(df)) * 100 if len(df) else 0
        if percentage == 0:
            action = "No Action Needed"
        elif 0 < percentage <= 5:
            action = "Drop The Rows"
        elif 5 < percentage <= 30:
            action = "Go for Imputation"
        else:
            action = "Drop the Column"
        report[col] = {
            "MISSING VALUE COUNT": int(df[col].isna().sum()),
            "DATATYPE": str(df[col].dtype),
            "PERCENTAGE": round(float(percentage), 2),
            "ACTION": action,
        }
    return pd.DataFrame.from_dict(report, orient="index")


def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='ISO8601', errors='coerce')
    return df


def apply_missing_value_strategy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    actions = {}
    rows_to_drop_mask = pd.Series(False, index=df.index)

    for col in list(df.columns):
        pct = (df[col].isna().sum() / len(df)) * 100 if len(df) else 0
        if pct == 0:
            actions[col] = "No Action Needed"
        elif 0 < pct <= 5:
            rows_to_drop_mask = rows_to_drop_mask | df[col].isna()
            actions[col] = "Rows marked for dropping"
        elif 5 < pct <= 30:
            actions[col] = "Imputed"
        else:
            df.drop(columns=[col], inplace=True)
            actions[col] = "Dropped Column"

    if rows_to_drop_mask.any():
        df = df.loc[~rows_to_drop_mask].copy()

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            imputer = SimpleImputer(strategy='median')
            df[col] = imputer.fit_transform(df[[col]]).ravel()
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # forward fill then back fill for timestamps if needed
            df[col] = df[col].ffill().bfill()
        else:
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel()

    return df, actions


def cap_outliers_iqr(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df_capped = dataframe.copy()
    applied = []
    for col in columns:
        if col not in df_capped.columns or not pd.api.types.is_numeric_dtype(df_capped[col]):
            continue
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
        applied.append(col)
    return df_capped


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        base = df['Timestamp'].min()
        df['timestamp_seconds'] = (df['Timestamp'] - base).dt.total_seconds()
        df['hour'] = df['Timestamp'].dt.hour
        df['minute'] = df['Timestamp'].dt.minute
        df['second'] = df['Timestamp'].dt.second
        df['dayofweek'] = df['Timestamp'].dt.dayofweek
    return df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = [c for c in ['Mode', 'Status'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


def prepare_modeling_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    data = df.copy().sort_values('Timestamp' if 'Timestamp' in df.columns else df.index.name or df.index)

    required_targets = [c for c in ['Latitude', 'Longitude', 'Altitude'] if c in data.columns]
    if len(required_targets) < 3:
        raise ValueError("Dataset must include Latitude, Longitude, and Altitude for trajectory prediction.")

    data = build_time_features(data)

    # Lag features from current trajectory state
    for col in required_targets + [c for c in ['Speed', 'Heading', 'Pitch', 'Roll', 'Yaw'] if c in data.columns]:
        data[f'{col}_lag1'] = data[col].shift(1)

    # Predict next-step path
    data['target_latitude_next'] = data['Latitude'].shift(-1)
    data['target_longitude_next'] = data['Longitude'].shift(-1)
    data['target_altitude_next'] = data['Altitude'].shift(-1)

    data = encode_categories(data)
    data = data.dropna().reset_index(drop=True)

    target_cols = ['target_latitude_next', 'target_longitude_next', 'target_altitude_next']

    excluded = {'Timestamp', 'Serial_No.'} | set(target_cols)
    feature_cols = [c for c in data.columns if c not in excluded and pd.api.types.is_numeric_dtype(data[c])]

    if not feature_cols:
        raise ValueError("No usable numeric feature columns were available after preprocessing.")

    return data, feature_cols, target_cols


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> ModelArtifacts:
    model_df, feature_cols, target_cols = prepare_modeling_dataframe(df)

    X = model_df[feature_cols]
    y = model_df[target_cols]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, model_df.index, test_size=test_size, random_state=random_state, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    pred_df = pd.DataFrame(preds, columns=target_cols, index=idx_test)

    actual = y_test.copy()
    result_df = model_df.loc[idx_test, ['Timestamp', 'Latitude', 'Longitude', 'Altitude']].copy() if 'Timestamp' in model_df.columns else model_df.loc[idx_test, ['Latitude', 'Longitude', 'Altitude']].copy()
    result_df['actual_next_latitude'] = actual['target_latitude_next'].values
    result_df['actual_next_longitude'] = actual['target_longitude_next'].values
    result_df['actual_next_altitude'] = actual['target_altitude_next'].values
    result_df['pred_next_latitude'] = pred_df['target_latitude_next'].values
    result_df['pred_next_longitude'] = pred_df['target_longitude_next'].values
    result_df['pred_next_altitude'] = pred_df['target_altitude_next'].values

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2 Score": float(r2_score(y_test, preds)),
        "Train Rows": int(len(X_train)),
        "Test Rows": int(len(X_test)),
        "Feature Count": int(len(feature_cols)),
    }

    return ModelArtifacts(
        feature_columns=feature_cols,
        target_columns=target_cols,
        model=model,
        metrics=metrics,
        result_df=result_df
    )


def plot_histograms(df: pd.DataFrame, columns: List[str], max_cols: int = 6):
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        st.info("No numeric columns available for distribution plots.")
        return
    selected = st.multiselect(
        "Choose columns for distribution plots",
        valid_cols,
        default=valid_cols[:min(max_cols, len(valid_cols))]
    )
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]
    for ax, col in zip(axes, selected):
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)


def plot_boxplots_before_after(original_df: pd.DataFrame, processed_df: pd.DataFrame, columns: List[str]):
    valid_cols = [c for c in columns if c in original_df.columns and c in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[c])]
    if not valid_cols:
        st.info("No selected outlier columns available for boxplot comparison.")
        return
    selected = st.multiselect(
        "Choose columns for outlier comparison",
        valid_cols,
        default=valid_cols[:min(4, len(valid_cols))],
        key="outlier_cols"
    )
    if not selected:
        return
    for col in selected:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].boxplot(original_df[col].dropna(), vert=False)
        axes[0].set_title(f'{col} - Before')
        axes[1].boxplot(processed_df[col].dropna(), vert=False)
        axes[1].set_title(f'{col} - After')
        plt.tight_layout()
        st.pyplot(fig)


def plot_time_series(df: pd.DataFrame):
    needed = [c for c in ['Timestamp', 'Latitude', 'Longitude', 'Altitude'] if c in df.columns]
    if len(needed) < 4:
        st.info("Timestamp/Latitude/Longitude/Altitude are needed for time-series plots.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(df['Timestamp'], df['Latitude'])
    axes[0].set_title('Time vs Latitude')
    axes[1].plot(df['Timestamp'], df['Longitude'])
    axes[1].set_title('Time vs Longitude')
    axes[2].plot(df['Timestamp'], df['Altitude'])
    axes[2].set_title('Time vs Altitude')
    for ax in axes:
        ax.grid(True)
        ax.set_xlabel('Timestamp')
    plt.tight_layout()
    st.pyplot(fig)


def plot_flight_path_2d(df: pd.DataFrame):
    if not all(c in df.columns for c in ['Latitude', 'Longitude']):
        st.info("Latitude and Longitude are required for 2D flight path plotting.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Longitude'], df['Latitude'], marker='o', linestyle='--')
    ax.set_title('2D Flight Path')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


def plot_flight_path_3d(df: pd.DataFrame):
    if not all(c in df.columns for c in ['Longitude', 'Latitude', 'Altitude']):
        st.info("Longitude, Latitude, and Altitude are required for 3D flight path plotting.")
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['Longitude'], df['Latitude'], df['Altitude'], linewidth=1, marker='o', markersize=3)
    ax.set_title('3D Flight Path Analysis')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')
    plt.tight_layout()
    st.pyplot(fig)


def plot_status_mode_behavior(df: pd.DataFrame):
    for group_col in ['Status', 'Mode']:
        if group_col in df.columns and all(c in df.columns for c in ['Speed', 'Altitude']):
            behavior = df.groupby(group_col)[['Speed', 'Altitude']].mean(numeric_only=True)
            st.subheader(f'Flight Behavior by {group_col}')
            st.dataframe(behavior)
            fig, ax = plt.subplots(figsize=(10, 5))
            behavior.plot(kind='bar', ax=ax)
            ax.set_ylabel('Average Value')
            ax.set_xlabel(group_col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)


def plot_predicted_vs_actual_path(results: pd.DataFrame):
    if not all(c in results.columns for c in ['actual_next_latitude', 'actual_next_longitude', 'pred_next_latitude', 'pred_next_longitude']):
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['actual_next_longitude'], results['actual_next_latitude'], label='Actual Path', marker='o')
    ax.plot(results['pred_next_longitude'], results['pred_next_latitude'], label='Predicted Path', marker='x')
    ax.set_title('Actual vs Predicted Flight Path (2D)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


def plot_predicted_vs_actual_3d(results: pd.DataFrame):
    if not all(c in results.columns for c in ['actual_next_longitude', 'actual_next_latitude', 'actual_next_altitude', 'pred_next_longitude', 'pred_next_latitude', 'pred_next_altitude']):
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(results['actual_next_longitude'], results['actual_next_latitude'], results['actual_next_altitude'], label='Actual', marker='o')
    ax.plot(results['pred_next_longitude'], results['pred_next_latitude'], results['pred_next_altitude'], label='Predicted', marker='x')
    ax.set_title('Actual vs Predicted Flight Path (3D)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def main():
    st.title("Drone Trajectory Analysis & Prediction System")
    st.caption("EDA dashboard, preprocessing pipeline, linear regression training, testing, evaluation, and flight path prediction visualization.")

    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload drone telemetry CSV", type=["csv"])
        rename_cols = st.checkbox("Rename columns using notebook mapping", value=True)
        apply_capping = st.checkbox("Apply IQR outlier capping on selected columns", value=True)
        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        st.markdown(
            """
            **Expected workflow in this app**
            1. Data upload  
            2. Missing-value check  
            3. Missing-value imputation using your notebook rules  
            4. Timestamp conversion  
            5. Distribution and outlier analysis  
            6. EDA dashboard  
            7. Linear regression training and testing  
            8. Flight path prediction and visualization
            """
        )
        return

    raw_df = load_uploaded_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    df = raw_df.copy()

    if rename_cols and len(df.columns) == len(DEFAULT_NEW_NAMES):
        df = rename_columns_safely(df, DEFAULT_NEW_NAMES)

    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Cells", int(df.isna().sum().sum()))

    st.subheader("Missing Value Report - Before Timestamp Conversion")
    before_report = missing_value_report(df)
    st.dataframe(before_report, use_container_width=True)

    df = convert_timestamp(df)

    st.subheader("Missing Value Report - After Timestamp Conversion")
    after_report = missing_value_report(df)
    st.dataframe(after_report, use_container_width=True)

    processed_df, actions = apply_missing_value_strategy(df)

    st.subheader("Preprocessing Action Summary")
    action_df = pd.DataFrame({"Column": list(actions.keys()), "Action Applied": list(actions.values())})
    st.dataframe(action_df, use_container_width=True)

    capped_df = cap_outliers_iqr(processed_df, DEFAULT_OUTLIER_COLUMNS) if apply_capping else processed_df.copy()

    tabs = st.tabs([
        "Data Health",
        "Distribution & Outliers",
        "EDA Dashboard",
        "Model Training & Evaluation",
        "Prediction"
    ])

    with tabs[0]:
        st.subheader("Processed Data Preview")
        st.dataframe(capped_df.head(10), use_container_width=True)
        buffer = io.StringIO()
        capped_df.info(buf=buffer)
        st.text(buffer.getvalue())
        st.subheader("Statistical Summary")
        st.dataframe(capped_df.describe(include='all').transpose(), use_container_width=True)

    with tabs[1]:
        st.subheader("Data Distribution Check")
        plot_histograms(capped_df, NUMERIC_DISTRIBUTION_COLUMNS)
        st.subheader("Outlier Detection and Handling")
        plot_boxplots_before_after(processed_df, capped_df, DEFAULT_OUTLIER_COLUMNS)

    with tabs[2]:
        st.subheader("Complete EDA Dashboard")
        plot_time_series(capped_df)
        col1, col2 = st.columns(2)
        with col1:
            plot_flight_path_2d(capped_df)
        with col2:
            plot_flight_path_3d(capped_df)
        plot_status_mode_behavior(capped_df)

    artifacts = None
    with tabs[3]:
        st.subheader("Predictive Modeling with Linear Regression")
        try:
            artifacts = train_and_evaluate(capped_df, test_size=test_size, random_state=int(random_state))
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{artifacts.metrics['MAE']:.6f}")
            m2.metric("RMSE", f"{artifacts.metrics['RMSE']:.6f}")
            m3.metric("R² Score", f"{artifacts.metrics['R2 Score']:.6f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Train Rows", artifacts.metrics["Train Rows"])
            c2.metric("Test Rows", artifacts.metrics["Test Rows"])
            c3.metric("Features Used", artifacts.metrics["Feature Count"])

            st.markdown("**Model feature columns**")
            st.write(artifacts.feature_columns)

            st.markdown("**Testing results sample**")
            st.dataframe(artifacts.result_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Model training could not be completed: {e}")

    with tabs[4]:
        st.subheader("Flight Path Prediction and Visualization")
        if artifacts is None:
            try:
                artifacts = train_and_evaluate(capped_df, test_size=test_size, random_state=int(random_state))
            except Exception as e:
                st.error(f"Prediction view unavailable: {e}")
                return

        st.dataframe(artifacts.result_df.head(50), use_container_width=True)
        plot_predicted_vs_actual_path(artifacts.result_df)
        plot_predicted_vs_actual_3d(artifacts.result_df)

        csv = artifacts.result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download prediction results CSV",
            data=csv,
            file_name="drone_prediction_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
