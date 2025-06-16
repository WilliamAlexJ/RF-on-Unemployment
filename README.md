# RF-on-Unemployment

This project explores how to model and predict the U.S. unemployment rate using macroeconomic indicators from the Federal Reserve Economic Database. I take a statistical approach to handle the challenges of time series data, such as autocorrelation, structural breaks, and the non i.i.d. nature of economic sequences. The goal is to combine traditional statistical rigor with machine learning methods, which in this case is Random Forest regression with lagged variables that is used to produce a robust and interpretable forecast model.

Data Sources and Motivation

I used publicly available macroeconomic indicators from FRED, including:

UNRATE: Civilian unemployment rate (target variable)
ICSA: Initial claims for unemployment insurance (weekly)
T10Y3M: Yield curve spread between 10-year and 3-month Treasury rates (daily)
INDPRO: Industrial production index (monthly)
PAYEMS: Total nonfarm payrolls (monthly)
UMCSENT: University of Michigan consumer sentiment index (monthly)
USREC: Binary indicator of official NBER recession dates (monthly)
These variables were selected because they have well-documented relationships with labor market conditions, economic activity, and consumer behavior.

Data Preprocessing and Wrangling

Since the data came at different temporal granularities (weekly, daily, monthly), the first step was time harmonization. I converted all time indices to "end-of-month" timestamps to ensure alignment. Non-monthly series like ICSA and T10Y3M were resampled to monthly frequency using their last recorded value each month.

After renaming columns and aligning all datasets, I merged them into a single DataFrame and dropped any rows containing missing values.

I also truncated the data to the period from January 2000 to December 2020. This decision was made to exclude the volatile COVID-19 pandemic period (2020–2022) from training to avoid distortion due to non-stationary behavior and extreme outliers. The main goal of this project is not to handle volatile environments.

Handling Time Series Properties

One of the key statistical challenges with economic time series is autocorrelation, the fact that current unemployment values are strongly dependent on recent past values. Since Random Forest is not inherently aware of time and assumes observations are i.i.d., I addressed this by explicitly creating lagged features:

UNRATE_lag1: unemployment rate 1 month ago
UNRATE_lag2: unemployment rate 2 months ago
UNRATE_lag3: unemployment rate 3 months ago
These were added to the feature matrix, allowing the model to learn temporal dependencies in the data.

Modeling Approach

I used Random Forest Regressor, which is a tree based ensemble model that is robust to multicollinearity, nonlinearity, and noise. Its ability to capture complex relationships between features without assuming linearity made it ideal for macroeconomic data.

I performed two types of train/test splits:

Random Split (80/20): Gave very high accuracy (R² ≈ 0.94, RMSE ≈ 0.43) but overestimates performance because it violates the temporal structure.
Chronological Split (80% train, 20% test by time): Gave more realistic results (R² ≈ 0.49, RMSE ≈ 1.75), revealing that the model’s ability to generalize to the future is limited but still informative.

Residual Diagnostics

To ensure statistical validity of the model, I conducted several residual diagnostics:

Residual plots showed no trend or structure, suggesting that the model captured most of the systematic variation.
![resplotRF](https://github.com/user-attachments/assets/43fa2ed6-02f3-4c58-b669-ca7c5c9a0822)


Histogram of residuals approximated a normal distribution.
![histogramRF](https://github.com/user-attachments/assets/cbb09831-9f04-44f7-b34c-1a1a4da192a1)


Ljung-Box test for autocorrelation (lag=10) on the residuals gave a p-value > 0.99, strongly supporting the hypothesis that the residuals are uncorrelated which is a key assumption for a good predictive model.

Key Insights and Statistical Value

Incorporating lagged unemployment values was crucial for modeling autocorrelation which is something often ignored in naive machine learning applications.
Avoiding the COVID period was statistically prudent, as that data represents a structural break not reflected in past relationships.
Feature engineering, time-aware validation, and diagnostic testing gave this model depth and credibility beyond black-box prediction.

Potential Improvements and Extensions

Model uncertainty could be captured using Quantile Regression Forests.
Comparison with simpler models like ARIMA, Ridge regression with lags would offer benchmark baselines.
Incorporating rolling forecasts could simulate deployment conditions more realistically.
Forecasts could be extended to post-2020 data after retraining or fine tuning on post-pandemic regimes.
