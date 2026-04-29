# Drone Trajectory Analysis & Prediction System

This project is a complete Streamlit application built from the uploaded notebook flow and extended with:

- CSV data upload
- Missing value report
- Missing value handling using the same decision logic from the notebook
- Timestamp conversion
- Data distribution plots
- Outlier detection and IQR capping
- Complete EDA dashboard
- Linear regression model training and testing
- Evaluation metrics: MAE, RMSE, R²
- Predicted flight path visualization in 2D and 3D

## Project files

- `app.py` - main Streamlit app
- `requirements.txt` - Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected data

The app expects a drone telemetry CSV.  
If the uploaded CSV has 27 columns, the notebook column renaming map will be applied automatically when the checkbox is enabled.

## Modeling logic

The app uses **Linear Regression** to predict the **next step trajectory**:
- next latitude
- next longitude
- next altitude

It builds time-based and lag-based features from the telemetry stream and compares predicted vs actual path.

## Notes

- The missing-value decision rules match the notebook:
  - 0%: no action
  - 0 to 5%: drop rows
  - 5 to 30%: impute
  - >30%: drop column
- Timestamp conversion follows the notebook approach using `pd.to_datetime(..., format='ISO8601', errors='coerce')`
- Outlier capping is applied to the same selected columns from the notebook by default
