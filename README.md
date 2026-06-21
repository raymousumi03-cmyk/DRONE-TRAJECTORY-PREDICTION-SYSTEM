# DRONE-TRAJECTORY-PREDICTION-SYSTEM
----
# Project Overview

This project focuses on Drone Trajectory Analysis and Prediction using drone telemetry data. An interactive web application was developed using Streamlit to perform data preprocessing, exploratory data analysis (EDA), machine learning-based trajectory prediction, and visualization of drone flight paths. The system helps analyze drone behavior and predict the drone's next geographical position based on historical flight data.

----

# Key Contributions

- Developed an interactive **Streamlit-based web application** for drone telemetry data analysis and trajectory prediction.
- Processed and standardized raw drone telemetry datasets through automated column renaming and preprocessing techniques.
- Performed **missing value detection and treatment** using data-driven strategies, including row removal, imputation, and column elimination.
- Converted timestamp data into **datetime format** for effective time-series analysis and feature extraction.
- Implemented **IQR-based outlier detection and capping** to improve data quality and enhance model performance.
- Conducted **Exploratory Data Analysis (EDA)** using statistical summaries, histograms, box plots, and distribution analysis.
- Visualized drone movement through **2D and 3D flight path plots** for comprehensive trajectory analysis.
- Analyzed flight behavior based on different **drone modes and operational statuses**.
- Engineered **time-based features** (hour, minute, second, day of week) and **lag-based features** to capture temporal flight patterns.
- Applied **One-Hot Encoding** to transform categorical variables into machine-learning-ready features.
- Built a **Linear Regression model** to predict future drone coordinates, including Latitude, Longitude, and Altitude.
- Evaluated model performance using industry-standard metrics such as **MAE (Mean Absolute Error)**, **RMSE (Root Mean Squared Error)**, and **R² Score**.
- Compared actual and predicted drone trajectories through interactive visualization techniques.
- Implemented **CSV export functionality** to enable downloading and further analysis of prediction results.
- Developed a complete **end-to-end machine learning pipeline**, covering data preprocessing, EDA, feature engineering, model training, evaluation, and trajectory prediction.

----

# Project Outcome
Successfully built a comprehensive Drone Trajectory Analysis and Prediction System capable of processing telemetry data, analyzing flight behavior, visualizing drone movements, and predicting future flight paths using machine learning techniques.


![Dashboard preview]("C:\Users\raymo\OneDrive\Pictures\Screenshots\Screenshot 2026-06-21 162820.png")
