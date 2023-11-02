
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from scipy.stats import zscore
from pykalman import KalmanFilter

def perform_analysis(data1, data2):
    # Align the data by date and perform cointegration test
    data = pd.concat([data1, data2], axis=1).dropna()
    data.columns = ['Asset1', 'Asset2']
    _, p_value, _ = coint(data['Asset1'], data['Asset2'])
    
    # Calculate the hedge ratio using Kalman Filter
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[data['Asset1']], [np.ones(len(data['Asset1']))]]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                      initial_state_mean=np.zeros(2),
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=1.0,
                      transition_covariance=trans_cov)

    state_means, _ = kf.filter(data['Asset2'].values)
    hedge_ratios = state_means[:, 0]
    data['Spread'] = data['Asset2'] - hedge_ratios * data['Asset1']

    # Calculate the z-score
    data['Z-Score'] = zscore(data['Spread'])
    
    return data, p_value, hedge_ratios

# Function to add the predicted closing prices to the data and re-calculate the z-score
def add_predicted_close(data, predicted_close1, predicted_close2, hedge_ratios):
    # Create a new DataFrame with the predicted close prices
    	new_row = pd.DataFrame({
        	'Asset1': [predicted_close1],
        	'Asset2': [predicted_close2]
    	}, index=[data.index[-1] + pd.Timedelta(days=1)])

    # Append the new row to the existing DataFrame
    	updated_data = pd.concat([data, new_row])
    
    # Recalculate the spread using the latest hedge ratio
    	latest_hedge_ratio = hedge_ratios[-1]
    	updated_data['Spread'] = updated_data['Asset2'] - latest_hedge_ratio * updated_data['Asset1']

    # Recalculate the z-score for the spread
    	updated_data['Z-Score'] = zscore(updated_data['Spread'])

    	return updated_data

# Title of the app
st.title('Financial Analysis and Trading Strategy App')

# Sidebar inputs for user to enter the tickers and date range
st.sidebar.header('User Input Features')
start_date = st.sidebar.date_input("Start date", pd.to_datetime('2022-06-01'))
end_date = st.sidebar.date_input("End date", pd.to_datetime('2023-11-02'))

# Ask for the user's predicted closing prices
#st.sidebar.header('Predicted Closing Prices for Today')
predicted_close_price1 = st.sidebar.number_input(f'Predict the closing price for ticker 1 ', value=float(100))
predicted_close_price2 = st.sidebar.number_input(f'Predict the closing price for ticker2', value=float(100))

# User input for tickers
ticker1 = st.sidebar.text_input('Enter first ticker symbol (e.g., QQQ)', 'QQQ')
ticker2 = st.sidebar.text_input('Enter second ticker symbol (e.g., SPY)', 'SPY')






# Placeholder for the main content
main_content = st.empty()

# Fetching the data based on user inputs
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)['Adj Close']
    return data

try:
    # Display a message while loading data
    with main_content.container():
        st.write('Loading data for {} and {}...'.format(ticker1, ticker2))
    data1 = load_data(ticker1, start_date, end_date)
    data2 = load_data(ticker2, start_date, end_date)
    # Display a message once data is loaded successfully
    with main_content.container():
        st.write('Data loaded successfully!')



finally:



# Display results with the predicted close prices
	if 'processed_data' in locals() and 'hedge_ratios' in locals():
    		updated_data_with_predicted_close = add_predicted_close(
        	processed_data, predicted_close_price1, predicted_close_price2, hedge_ratios)
    
    # Display the updated z-score for the user-provided closing prices
    		st.write(f'Updated z-score with predicted closing prices: {updated_data_with_predicted_close["Z-Score"].iloc[-1]:.4f}')
#except Exception as e:
    # Display an error message if data loading fails
 #   		with main_content.container():
  #      		st.error('An error occurred while fetching the data. Please check the ticker symbols and try again.')
   #     		st.exception(e)



# Display results
if 'data1' in locals() and 'data2' in locals():
    processed_data, p_value, hedge_ratios = perform_analysis(data1, data2)
    
    # Display the cointegration test results
    st.write(f'Cointegration test p-value: {p_value:.4f}')
    st.table(processed_data.tail(10))
    
    # Plot the spread and z-score
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(processed_data.index, processed_data['Spread'], label='Spread')
    ax[0].set_title('Spread over Time')
    ax[1].plot(processed_data.index, processed_data['Z-Score'], label='Z-Score', color='orange')
    ax[1].axhline(y=0, linestyle='--', color='black', lw=0.5)
    ax[1].set_title('Z-Score over Time')
    st.pyplot(fig)


