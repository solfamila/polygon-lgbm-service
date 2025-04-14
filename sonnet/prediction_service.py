import tensorflow as tf
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import time
from datetime import datetime, timedelta
import threading
import os
from dotenv import load_dotenv

PREDICTION_THRESHOLD = 0.6 # Or your desired threshold
TARGET_GAIN = 1.0        # Or your defined target gain
LOOKAHEAD_PERIOD = 15    # Define this constant (used in feature eng function default)

# Load models and associated artifacts
print("Loading models and artifacts...")
try:
    # RandomForest artifacts
    rf_model, rf_scaler, rf_feature_cols = joblib.load('randomforest_basic_model.joblib')
    print("RandomForest model and scaler loaded.")

    # LSTM artifacts (Ensure paths match where they were saved)
    lstm_model = tf.keras.models.load_model('lstm_model_tsla.keras')
    lstm_scaler, lstm_feature_cols = joblib.load('lstm_scaler_columns_tsla.joblib')
    # SEQ_LENGTH is needed from training phase - define it here or load it
    SEQ_LENGTH = 60 # Make sure this matches the sequence length used for LSTM training
    print("LSTM model, scaler, and feature columns loaded.")
    
except FileNotFoundError as e:
     print(f"Error: Model file not found - {e}. Please ensure models are saved in the correct path.")
     exit(1)
except Exception as e:
     print(f"Error loading models/artifacts: {e}")
     exit(1)
print("Models loaded successfully")

load_dotenv() # Load environment variables from .env file
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass") 

conn = psycopg2.connect(
    host="localhost",
    port="5433",       # <-- Add host port
    database="polygondata", # <-- Correct DB name
    user="polygonuser",     # <-- Correct user
    password=DB_PASS     # <-- Use loaded/correct password
)
conn.autocommit = True
cursor = conn.cursor()

# Create predictions table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS stock_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    current_price NUMERIC(10, 2) NOT NULL,
    rf_probability NUMERIC(10, 4),
    lstm_probability NUMERIC(10, 4),
    ensemble_probability NUMERIC(10, 4),
    trade_signal BOOLEAN,
    target_price NUMERIC(10, 2),
    stop_loss NUMERIC(10, 2)
);
""")
print("Prediction table created/verified")

# Function to get the most recent data for predictions
# Change this line:
# def get_recent_data(ticker='TSLA', minutes=120):

# To this:
def get_recent_data(ticker='TSLA', minutes=120, limit=200): # Added limit with default
    print(f"Fetching recent data for {ticker} (target: last {minutes} mins, DB limit {limit} rows)") # Updated print
    # Construct the interval string safely
    interval_string = f"'{int(minutes)} minutes'"

    # Query using the correct columns and interval
    query = f"""
    SELECT 
        start_time AS time, 
        agg_open   AS open,  
        agg_high   AS high,  
        agg_low    AS low,   
        agg_close  AS close, 
        volume,
        vwap,
        num_trades      
    FROM stock_aggregates_min
    WHERE 
        symbol = %(ticker)s 
        AND start_time > (NOW() AT TIME ZONE 'UTC' - INTERVAL {interval_string})
    ORDER BY start_time ASC; 
    """

    try:
        # Note: The primary query doesn't use the limit directly, 
        # it relies on the time filter. Limit is used in fallback.
        df = pd.read_sql(query, conn, params={'ticker': ticker}, index_col='time', parse_dates={'time': {'utc': True}})
        df.sort_index(ascending=True, inplace=True) 

        # Check if we got enough data with the time filter
        if len(df) < 60: 
            print(f"Warning: Only {len(df)} recent data points found in the last {minutes} minutes. Fetching last {limit} points regardless of time.")
            
            # Fallback query uses the limit parameter
            fallback_query = """
            SELECT 
                start_time AS time, 
                agg_open   AS open,  
                agg_high   AS high,  
                agg_low    AS low,   
                agg_close  AS close, 
                volume,
                vwap,
                num_trades      
            FROM stock_aggregates_min
            WHERE symbol = %(ticker)s
            ORDER BY start_time DESC 
            LIMIT %(limit)s; -- Use limit parameter here
            """
            df = pd.read_sql(fallback_query, conn, params={'ticker': ticker, 'limit': limit}, index_col='time', parse_dates={'time': {'utc': True}})
            df.sort_index(ascending=True, inplace=True) # Sort back to ascending

        print(f"Returning {len(df)} data points for prediction.")
        return df

    except Exception as e:
        print(f"Error fetching recent data: {e}")
        return pd.DataFrame() 


# Create features function (same as before)
def create_features(df, lookahead=LOOKAHEAD_PERIOD, gain=TARGET_GAIN):
    # Make sure DataFrame has enough rows
    min_required_rows = 60 + lookahead # Adjusted min requirement
    if len(df) < min_required_rows:
        print(f"Warning: DataFrame has {len(df)} rows, need at least {min_required_rows} for feature engineering.")
        return pd.DataFrame() 

    df = df.copy()
    
    # Basic Price/Volume features
    df['return'] = df['close'].pct_change()
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['log_volume'] = np.log1p(df['volume']) # Log transform volume
    df['price_vwap_diff'] = df['close'] - df['vwap'] if 'vwap' in df.columns else 0 # Check if vwap exists

    # Moving averages
    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = df['close'].rolling(window=window, min_periods=window).mean()
        df[f'ma_vol_{window}'] = df['volume'].rolling(window=window, min_periods=window).mean() # Volume MA
        df[f'close_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}'] # Price relative to MA

    # Momentum / Rate of Change (ROC)
    for window in [1, 5, 10, 20]:
        df[f'roc_{window}'] = df['close'].pct_change(periods=window)

    # Volatility (Std Dev of returns)
    for window in [10, 20]:
        df[f'volatility_{window}'] = df['return'].rolling(window=window, min_periods=window).std() * np.sqrt(window) # Annualize slightly?

    # RSI (Relative Strength Index) - Example using pandas-ta (requires install: pip install pandas-ta)
    try:
        import pandas_ta as ta
        df.ta.rsi(length=14, append=True) # Appends 'RSI_14' column
        print("Added RSI feature.")
    except ImportError:
        print("Warning: pandas_ta not installed. Skipping RSI feature. (pip install pandas-ta)")
        df['RSI_14'] = 0.5 # Placeholder if needed, or handle later
    except Exception as e:
        print(f"Warning: Error calculating RSI: {e}")
        df['RSI_14'] = 0.5 # Placeholder

    # Target Definition
    # df['future_price'] = df['close'].shift(-lookahead)
    # df['target'] = ((df['future_price'] - df['close']) >= gain).astype(int)
    
    # Drop rows with NaNs created by feature engineering
    print(f"Rows before dropna: {len(df)}")
    df.dropna(inplace=True)
    print(f"Rows after dropna: {len(df)}")
    
    return df

# Function to make predictions
# Function to make predictions
def make_prediction(ticker='TSLA'):
    print(f"\n--- Attempting Prediction for {ticker} at {datetime.now()} ---")
    try:
        # 1. Get recent data 
        recent_data_raw = get_recent_data(ticker, minutes=180, limit=300) 
        
        if recent_data_raw.empty or len(recent_data_raw) < 60 + SEQ_LENGTH: 
            print(f"Not enough data points fetched: {len(recent_data_raw)} (required ~{60 + SEQ_LENGTH})")
            return None
            
        # 2. Create features
        feature_df_full = create_features(recent_data_raw) 
        
        if feature_df_full.empty:
             print("Feature generation resulted in empty DataFrame.")
             return None
        
        # 3. Get the latest row
        latest_complete_row = feature_df_full.iloc[-1:] 

        if latest_complete_row.empty:
            print("Could not get latest row after feature calculation.")
            return None

        current_price = latest_complete_row['close'].iloc[0]
        current_time = latest_complete_row.index[0] 

        print(f"Latest data point time: {current_time}, Price: {current_price}")

        # 4. RandomForest Prediction (Using Correct Columns)
        print("Preparing features for RandomForest...")
        rf_prob = 0.0 # Default value
        try:
            # Ensure the necessary columns for RF exist in the latest row before selecting
            required_rf_cols_exist = all(col in latest_complete_row.columns for col in rf_feature_cols)
            
            if not required_rf_cols_exist:
                print(f"Warning: Missing columns for RF prediction in latest data point. RF Features required: {rf_feature_cols}")
                print(f"Available columns: {latest_complete_row.columns.tolist()}")
            else:
                # Select ONLY the specific columns RF was trained on
                rf_features_predict = latest_complete_row[rf_feature_cols] 
            
                # Handle NaNs just in case
                rf_features_predict = rf_features_predict.fillna(0) 

                rf_features_scaled = rf_scaler.transform(rf_features_predict.values) 
                rf_prob = rf_model.predict_proba(rf_features_scaled)[0, 1]
                print(f"RF Prediction successful (Prob: {rf_prob:.4f})")

        except Exception as e:
            print(f"Error during RandomForest feature prep/prediction: {e}")
            # Keep rf_prob as 0.0

        # 5. LSTM Prediction (Using Correct Columns)
        print("Preparing features for LSTM...")
        lstm_prob = 0.0 # Default value
        try:
            # Ensure the necessary columns for LSTM exist before slicing
            required_lstm_cols_exist = all(col in feature_df_full.columns for col in lstm_feature_cols)
            if not required_lstm_cols_exist:
                print(f"Warning: Missing columns for LSTM prediction in feature data. LSTM Features required: {lstm_feature_cols}")
            else:
                # Select sequence data using ONLY the features LSTM expects
                lstm_sequence_data = feature_df_full[lstm_feature_cols].iloc[-SEQ_LENGTH:].values

                if lstm_sequence_data.shape[0] < SEQ_LENGTH:
                     print(f"Not enough data rows ({lstm_sequence_data.shape[0]}) for LSTM sequence length {SEQ_LENGTH}")
                else:
                     lstm_sequence_data = np.nan_to_num(lstm_sequence_data, nan=0.0) 
                     lstm_features_scaled = lstm_scaler.transform(lstm_sequence_data) 
                     lstm_seq_input = lstm_features_scaled.reshape(1, SEQ_LENGTH, len(lstm_feature_cols)) 
                     lstm_prob = lstm_model.predict(lstm_seq_input, verbose=0)[0, 0]
                     print(f"LSTM Prediction successful (Prob: {lstm_prob:.4f})")

        except Exception as e:
             print(f"Error during LSTM feature prep/prediction: {e}")
             # Keep lstm_prob as 0.0
        
        # 6. Ensemble, Signal, Store
        ensemble_prob = (rf_prob + lstm_prob) / 2
        trade_signal = ensemble_prob > PREDICTION_THRESHOLD 
        target_price = current_price + TARGET_GAIN  
        stop_loss = current_price - (TARGET_GAIN / 2) 

        # Store prediction in database (with type casting)
        insert_query = """
        INSERT INTO stock_predictions 
        (timestamp, ticker, current_price, rf_probability, lstm_probability, 
        ensemble_probability, trade_signal, target_price, stop_loss)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        insert_data = (
            current_time.to_pydatetime(), 
            ticker, 
            float(current_price), # <--- FIX: Cast current_price to float
            float(rf_prob),       
            float(lstm_prob),     
            float(ensemble_prob), 
            bool(trade_signal),   # Cast boolean
            float(target_price) if bool(trade_signal) else None, # Cast target/stop too
            float(stop_loss) if bool(trade_signal) else None    
        )

        try:
             cursor.execute(insert_query, insert_data)
             print("Prediction saved to database.")
        except Exception as db_err:
             print(f"Error saving prediction to DB: {db_err}")
             print("Data that failed insertion:", insert_data) # Print data for debugging
        
        # Print prediction details (even if save failed)
        print(f"--- Prediction Summary ---")
        print(f"Time: {current_time}")
        # ... (rest of print statements) ...
        
        return insert_data # Or return a dictionary
        
    except Exception as e:
        print(f"Error in make_prediction function: {e}")
        import traceback
        traceback.print_exc() 
        return None




# Function to run predictions on a schedule
def run_prediction_service(interval_seconds=60):
    while True:
        try:
            # Make prediction
            prediction = make_prediction('TSLA')
            
            # Sleep until next interval
            time.sleep(interval_seconds)
            
        except Exception as e:
            print(f"Prediction service error: {e}")
            time.sleep(5)  # Short sleep on error

# Start the prediction service in a background thread
print("Starting prediction service...")
prediction_thread = threading.Thread(target=run_prediction_service, daemon=True)
prediction_thread.start()

# Main thread - keep alive and provide simple interface
try:
    print("\nPrediction service running. Press Ctrl+C to exit.")
    print("Recent predictions:")
    
    while True:
        # Display recent predictions
        cursor.execute("""
        SELECT timestamp, ticker, current_price, ensemble_probability, trade_signal
        FROM stock_predictions
        ORDER BY timestamp DESC
        LIMIT 5;
        """)
        
        recent_preds = cursor.fetchall()
        if recent_preds:
            print("\nRecent predictions:")
            for pred in recent_preds:
                signal = "BUY" if pred[4] else "HOLD"
                print(f"{pred[0]} | {pred[1]} | ${pred[2]:.2f} | {pred[3]:.4f} | {signal}")
        
        time.sleep(10)
        
except KeyboardInterrupt:
    print("\nShutting down prediction service...")
    conn.close()
    print("Done.")

