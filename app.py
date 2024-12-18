import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained LSTM model and scaler
with open('lstm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title('Stock Price Prediction using LSTM')

# Upload the CSV file via Streamlit UI
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV data
    data = pd.read_csv(uploaded_file)
    st.write(data.tail())  # Show the last 5 rows of the dataset

    # Process the closing price for LSTM
    closing_price = data[['Close']].values
    scaled_closing_price = scaler.transform(closing_price)

    # Prepare the dataset for LSTM
    time_step = 60
    X = []
    for i in range(len(scaled_closing_price) - time_step):
        X.append(scaled_closing_price[i:i + time_step, 0])

    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Predict the next 30 days
    predictions = model.predict(X[-30:])

    # Rescale the predicted prices
    predicted_prices = scaler.inverse_transform(predictions)

    # Display predicted stock prices
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Calculate predicted dates correctly
    predicted_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30).strftime('%Y-%m-%d')

    # Display prediction results
    st.subheader("Predicted Stock Prices for the Next 30 Days")
    predicted_df = pd.DataFrame(predicted_prices, columns=["Predicted Close"], index=predicted_dates)
    st.write(predicted_df)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_df.index, predicted_df['Predicted Close'], label="Predicted Price", color="orange")
    plt.title('Predicted Stock Prices (Next 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
