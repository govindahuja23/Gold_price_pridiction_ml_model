import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import random
import time

# -------------------------
# Page Config
# -------------------------
st.header('Gold Price Prediction Using Machine Learning')

desc = '''
Gold Price Prediction using Machine Learning.  
Gold price forecasting is crucial for investors and financial markets. 
Machine Learning can analyze multiple economic indicators and market indices to predict the **future closing price of Gold**.  

In this project, an XGBoost regression model was trained on gold price dataset 
with features like commodity prices, currency indices, oil prices, and stock indices.


Algorithms Used:

**Linear Regression**  
**Decision Tree Regressor**  
**Random Forest Regressor**  
**XGBoost Regressor**  


# -------------------------
# Column Descriptions Section
# -------------------------
st.markdown("## 📊 Column Descriptions for Gold Price Prediction")

st.markdown("""
| **Column**    | **Description** |
|---------------|-----------------|
| **Date**      | Trading date. |
| **Open**      | Gold opening price on that day. |
| **High**      | Highest gold price during the trading session. |
| **Low**       | Lowest gold price during the trading session. |
| **Adj Close** | Adjusted closing price of gold (accounts for splits, dividends). |
| **Volume**    | Trading volume (number of transactions/shares/contracts traded). |
| **SP_close**  | S&P 500 Index closing value (market performance indicator). |
| **DJ_close**  | Dow Jones Industrial Average closing value (market benchmark). |
| **EG_close**  | Euro Gold Index closing value. |
| **USDI_Price**| USD Index (measures USD strength vs major currencies). |
| **EU_Price**  | Euro Index (measures Euro’s relative value). |
| **SF_Price**  | Swiss Franc Index (measures CHF value). |
| **PLT_Price** | Platinum price (precious metal market correlation). |
| **PLD_Price** | Palladium price (precious metal market correlation). |
| **USO_Close** | U.S. Oil Fund closing value (energy market factor). |
| **GDX_Close** | Gold Miners ETF closing value (mining stock performance). |
| **RHO_PRICE** | Rhodium price (another precious metal, supply-demand factor). |
| **Close**     | Gold closing price (**Target variable** for prediction). |
""")

'''

st.markdown(desc)

st.image("https://th.bing.com/th/id/OIP.M-h4V94xqn8LkQBG1RmnjgHaFS?w=265&h=189&c=7&r=0&o=7&pid=1.7&rm=3")

# -------------------------
# Load trained model & scaler
# -------------------------
model = joblib.load("/mount/src/gold_price_pridiction_ml_model/Gold/xgb_model.pkl")
import os
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler = joblib.load(scaler_path)

  # <<-- load fitted scaler

# -------------------------
# Load dataset (for slider ranges)
# -------------------------

csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
df = pd.read_csv(csv_path)

# The features used for training (excluding Close)
feature_cols = ['Volume', 'SP_close', 'DJ_close', 'EG_close', 'USDI_Price',
                'EU_Price', 'SF_Price', 'PLT_Price', 'PLD_Price', 'USO_Close',
                'GDX_Close', 'RHO_PRICE']

st.sidebar.header('Select Features to Predict Gold Price')
st.sidebar.image('https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXp4Ym53dGh5NXU3bGNwYWI0a3lkdGw2bTMyc3Z4cWF6NDJpZTJ1biZlcD12MV9naWZzX3NlYXJjaCZjdD1n/l4FGnZ5NlHuvHfthm/giphy.webp', width=200)

# -------------------------
# Collect input values
# -------------------------
all_values = []
for col in feature_cols:
    min_value, max_value = df[col].agg(['min','max'])
    val = st.sidebar.slider(f'Select {col}', float(min_value), float(max_value),
                            float(random.uniform(min_value, max_value)))
    all_values.append(val)

# Convert to correct format & scale
final_value = np.array([all_values], dtype=float)
final_value_scaled = scaler.transform(final_value)   # <<-- scale inputs

# -------------------------
# Prediction with progress bar
# -------------------------
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Gold Price...')

place = st.empty()
place.image('https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXp4Ym53dGh5NXU3bGNwYWI0a3lkdGw2bTMyc3Z4cWF6NDJpZTJ1biZlcD12MV9naWZzX3NlYXJjaCZjdD1n/pPzjpxJXa0pna/200.webp', width=200)

random.seed(12)
for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)

prediction = model.predict(final_value_scaled)[0]

# -------------------------
# Display Result
# -------------------------
placeholder.empty()
place.empty()
progress_bar.empty()

st.subheader("📈 Predicted Gold Closing Price")
st.success(f"Estimated Gold Price: **₹{prediction:,.2f}**")


