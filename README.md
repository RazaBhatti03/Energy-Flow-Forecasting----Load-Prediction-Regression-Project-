#  Energy Flow Forecasting — Zone 1 Load Prediction (Regression Project)

Forecasting energy flow accurately is essential for optimizing power distribution, reducing energy waste, and planning future infrastructure. In this project, we build a robust regression model to predict **Zone 1 energy flow** using environmental and time-based factors such as temperature, humidity, time, and diffuse solar flows.

---

##  Objective

- This regression model forecasts energy usage in **Zone 1** based on environmental and time-based factors such as temperature, humidity, solar flows, and time features (hour, day, month).
- Perform complete EDA, feature engineering, and multivariate analysis.
- Improve model accuracy using hyperparameter tuning and ensemble learning.
- Handle time-related and nonlinear relationships in the dataset.

---

##  Business Problem

Energy demand forecasting helps:
- Utility companies optimize power generation and prevent overloads.
- Governments and industries reduce energy waste and operational cost.
- Engineers and planners design efficient smart grids.

---

##  Dataset Description

| Feature                | Description                                           |
|------------------------|-------------------------------------------------------|
| Temperature            | Ambient air temperature (°C)                          |
| Humidity               | Relative humidity (%)                                 |
| Wind Speed             | Wind speed (m/s)                                      |
| General Diffuse Flows  | Solar energy diffused over general surface (W/m²)     |
| Diffuse Flows          | Direct diffuse solar energy (W/m²)                    |
| DateTime               | Timestamp of record (10-minute intervals)             |
| Zone 1 (Target)        | Energy flow in Zone 1 (Watts)                         |

**Extracted Features:**
- `hour`, `dayofweek`, `month`, `is_weekend`

---

##  Steps Performed

###  1. Data Cleaning & Preprocessing
- Handled mixed datetime formats.
- Removed outliers using IQR method.
- Converted `DateTime` into multiple time-based features.

###  2. Exploratory Data Analysis (EDA)
- Univariate, bivariate, and multivariate visualizations.
- Outlier and distribution analysis using histograms, boxplots, pairplots.
- Time-based usage patterns extracted using heatmaps.

###  3. Feature Engineering
- Extracted `hour`, `dayofweek`, `month`, `is_weekend` from datetime.
- Removed low-importance features based on:
  - RandomForest feature importance
  - Mutual Information scores

###  4. Model Building & Evaluation
- Tried multiple regression models:
  - Linear Regression, Decision Tree, Random Forest, XGBoost
- Evaluated using:
  - R² Score, RMSE, MAE on train/test sets
- Used `GridSearchCV` for hyperparameter tuning

###  5. Final Model & Deployment Readiness
- Best model: `RandomForestRegressor` (R² ≈ 94% on test set)
- Saved model pipeline using `joblib`
- Ready to deploy on new data with consistent preprocessing

---

##  Key Visualizations

| Plot Type            | Purpose                                           |
|----------------------|---------------------------------------------------|
| Heatmap              | Feature correlation insight                       |
| Pairplot             | Nonlinear feature-target relationships            |
| Boxplot by Hour/Day  | Time-based peak usage visualization               |
| 3D Scatter Plot      | Combined effect of Temp, Humidity on Zone 1       |

> All plots are included in the `visualizations/` folder.

---

##  Tech Stack

- **Language**: Python 3.10  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost  
- **Tool**: Jupyter Notebook, Visual Studio Code  
- **Model Saving**: `joblib`  
- **Project Format**: Jupyter + Script-based

---

##  Folder Structure

├── data/
│ ├── raw_data.csv
│ └── cleaned_data.csv
├── notebooks/
│ └── EDA_Modeling.ipynb
├── models/
│ └── final_model.pkl
├── visualizations/
│ └── heatmap.png, boxplots.png, ...
├── README.md
└── requirements.txt

---

## Results & Performance

Below is the comparison of model performance **before and after feature engineering & hyperparameter tuning**:

| Metric                | Baseline Model: Decision Tree | Final Model: Random Forest |
|------------------------|------------------------------|-----------------------------|
| R² Score (Train)       | 100.00%                      | 99.60%                     |
| R² Score (Test)        | 94.01%                       | 97.28%                     |
| Mean Squared Error     | 3,039,939.14                 | 1,381,976.13               |
| Mean Absolute Error    | 1,003.55                     | 787.86                     |
| R² Score (as float)    | 0.940                        | 0.973                      |
| Top Features           | `diffuse flows`, `general diffuse flows`, `Temperature` | Same (confirmed via model) |

### Key Improvements:
- R² score on test data improved from **94.01% → 97.28%**
- MSE reduced by **~54.5%**, showing significantly lower prediction error
- MAE dropped from **1,003.55 to 787.86**, indicating more accurate predictions
- Final model is more generalized and avoids overfitting (better test vs train match)


---

### Business Impact:
- **Efficiency**: Accurate forecasting helps optimize energy generation and reduce excess production.
- **Cost Saving**: Prevents energy overuse and load imbalance, lowering grid operation costs.
- **Sustainability**: Supports smart grid systems by aligning energy demand with environmental conditions.
- **Scalability**: The framework can be extended to predict usage in **Zone 2 and Zone 3**, or for **real-time monitoring using IoT feeds**.

---

## Conclusion

This project demonstrates the full pipeline of a real-world machine learning solution:
- Data cleaning, outlier handling, and time-series-aware feature engineering
- Comparison of multiple models (Decision Tree vs. Random Forest)
- Final model improved accuracy from **94.01% to 97.28% (R²)** and cut error metrics by nearly half

**Key Insight**: Solar flows (`diffuse flows`, `general diffuse flows`) and temperature are the most critical drivers of energy usage in Zone 1.

- This model is ready for integration into **energy management systems** or deployment as a **backend service** for energy dashboards.

---

## Future Work

- Deploy model via Flask or FastAPI
- Add real-time dashboard with Streamlit
- Integrate with IoT sensor data for live predictions
- Extend to Zones 2 and 3 (multi-output regression)

---

## Contact & Credits

**Author**: *Raza Ur Rahman*  
**Email**: razabhatti03@gmail.com  
**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---


