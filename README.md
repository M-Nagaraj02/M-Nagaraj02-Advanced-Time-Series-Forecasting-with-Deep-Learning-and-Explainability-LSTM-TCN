---
# ⚡ Advanced Time Series Forecasting using LSTM & TCN with SHAP
---
## 📌 Overview

This project implements an advanced Multivariate Time Series Forecasting System using:

* 🔁 LSTM (Long Short-Term Memory)
* 📡 TCN (Temporal Convolutional Network)
* 📊 SHAP Explainability
* 🔄 Rolling Forecast Validation

The objective is to compare deep learning architectures for forecasting while ensuring interpretability and robust validation.

---
## 📂 Dataset

Source: Public time series dataset
```python
https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption_days.csv
```
**Features:**

* Global Active Power
* Global Reactive Power
* Voltage
* Global Intensity
* Sub-metering Variables

---
🛠️ Technologies & Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

```
---
## 🔎 Exploratory Data Analysis (EDA)

Performed:

* Missing Value Check
* Multi-line Time Series Visualization
* Correlation Heatmap

---
## ⚙ Feature Scaling

```python
MinMaxScaler()
```

---
## 🔄 Sequence Generation

Converted time series into supervised learning format.

* Sequence Length = 30
* Sliding window approach
* Predict next timestep

**Output Shape:**

* X → (samples, 30, features)
* y → (samples,)
---
# 🧠 Models Implemented
---
## 1️⃣ LSTM Model

Built using TensorFlow Sequential API.

**Architecture:**

* LSTM (64 units, return_sequences=True)
* Dropout (0.2)
* LSTM (32 units)
* Dropout (0.2)
* Dense (16, ReLU)
* Dense (1 Output)

**Training Strategy:**

* Epochs: 50
* Batch Size: 32
* EarlyStopping (patience=5)
* Validation Split: 20%

✔ Prevents Overfitting
✔ Restores Best Weights

---
## 2️⃣ TCN Model (Temporal CNN)

Temporal Convolutional Network built using:

* Conv1D (Causal Padding)
* Dilation Rates (1 & 2)
* Batch Normalization
* ReLU Activation
* Global Average Pooling
* Dense Layers

---
## 📊 Model Evaluation

Metrics Used:

* RMSE
* MAE
* MAPE

Example Output:
```python
LSTM -> RMSE: ---- MAE: ---- MAPE: ----
TCN  -> RMSE: ---- MAE: ---- MAPE: ----
```
✔ Performance comparison between architectures
✔ Error quantification

---
## 🔄 Rolling Forecast Validation

Performed step-by-step rolling prediction:

* Simulates real-time forecasting
* Avoids look-ahead bias
* Evaluates temporal robustness

Output:
```python
Rolling Forecast RMSE: ----
```
---
## 🔍 Model Explainability using SHAP

**Used SHAP GradientExplainer.**

Explainability Includes:

* SHAP Summary Plot
* Feature Impact Visualization
* Global Feature Importance

✔ Makes Deep Learning Interpretable
✔ Identifies most influential time series variables

---
## 📈 Visualization

Generated:

* Time Series Plots
* Correlation Heatmap
* Training Curves
* Actual vs LSTM vs TCN Forecast Plot

---
## 💾 Model Saving

Saved trained models:
```python
lstm_timeseries_model.h5
tcn_timeseries_model.h5
```
---
## 🧠 Key Concepts Demonstrated

* Multivariate Time Series Forecasting
* LSTM Deep Learning
* Temporal Convolutional Networks
* Sequence Modeling
* EarlyStopping Regularization
* Rolling Forecast Backtesting
* Model Explainability (XAI)
* Performance Benchmarking

---
## 📂 Project Structure
``` python
Advanced-TimeSeries-LSTM-TCN/
│── timeseries_forecasting.ipynb
│── lstm_timeseries_model.h5
│── tcn_timeseries_model.h5
│── README.md
```
---
## 🎯 Project Highlights

* Dual Architecture Comparison (LSTM vs TCN)
* Rolling Forecast Validation Implemented
* SHAP Explainability Integrated
* EarlyStopping Optimization
* Production-Ready Model Export

---
## 👨‍💻 Author

**Nagaraj M**

GitHub: https://github.com/M-Nagaraj02

---
