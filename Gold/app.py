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
'''

st.markdown(desc)

st.image("https://th.bing.com/th/id/OIP.M-h4V94xqn8LkQBG1RmnjgHaFS?w=265&h=189&c=7&r=0&o=7&pid=1.7&rm=3")

# -------------------------
# Load trained model & scaler
# -------------------------
model = joblib.load("Gold/xgb_model.pkl")


# -------------------------
# Load dataset (for slider ranges)
# -------------------------
url = '''data.csv'''  # <-- replace with your dataset if online
df = pd.read_csv(url)

# The features used for training (excluding Close)
feature_cols = ['Volume', 'SP_close', 'DJ_close', 'EG_close', 'USDI_Price',
                'EU_Price', 'SF_Price', 'PLT_Price', 'PLD_Price', 'USO_Close',
                'GDX_Close', 'RHO_PRICE']

st.sidebar.header('Select Features to Predict Gold Price')
st.sidebar.image('https://media.tenor.com/_XAKt0yDGLIAAAAC/gold-bars.gif', width=200)

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
place.image('https://i.makeagif.com/media/1-17-2024/dw-jXM.gif', width=200)

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
st.success(f"Estimated Gold Price: **${prediction:,.2f}**")


