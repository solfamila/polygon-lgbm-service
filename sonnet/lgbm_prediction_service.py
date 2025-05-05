# --- Necessary Imports ---
import lightgbm as lgb # <-- Import LightGBM
import psycopg2
import pandas as pd
import numpy as np
import time
import os
import math 
from datetime import datetime, timedelta, timezone 
from dotenv import load_dotenv
import threading
import joblib 
from sklearn.preprocessing import StandardScaler

# --- Technical Analysis Library (Optional but good if features need it) ---
try:
    import pandas_ta as ta; PANDAS_TA_AVAILABLE = True; print("pandas_ta imported.")
except ImportError: PANDAS_TA_AVAILABLE = False; print("Warning: pandas_ta not installed.")

# --- Suppress specific known Pandas FutureWarning ---
import warnings
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.*")


print("\n--- LightGBM Real-Time Prediction Service ---")

# --- Configuration Loading & Constants ---
load_dotenv(); DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST="localhost"; DB_PORT="5433"; DB_NAME="polygondata"; DB_USER="polygonuser"
TICKER = "TSLA"
# --- Parameters needed from training/backtest ---
LOOKAHEAD_PERIOD = 15 # Same as used in training
TARGET_GAIN = 1.0     # Same gain definition used in training's target
# *** Choose threshold based on Walk-Forward Backtest analysis ***
PREDICTION_THRESHOLD = 0.60 # Example: Baseline threshold
FETCH_MINUTES = 180   # How far back to look for data history for features
FETCH_LIMIT = 500     # Max number of rows for fallback query (needs to be >= Feature Lookback + a buffer)
PREDICTION_INTERVAL_SECONDS = 60 # How often to make predictions

# Define features to exclude (MUST match the list used when saving artifacts)
# Add 'regime' if it was present during feature engineering used for model training
EXCLUDE_FROM_FEATURES = ['target_binary', 'price_change_pct', 'movement_class', 
                         'future_price', 'open', 'high', 'low', 'close', 'volume', 
                         'vwap', 'num_trades', 'hour', 'dayofweek', 'regime'] # Assuming 'regime' might be present

# Paths to saved artifacts (Make sure these match the saved files from training)
MODEL_ARTIFACT_PATH = f'lgbm_final_model_enhanced_{TICKER.lower()}.joblib' # Load the LightGBM final model

# --- Load Model and Artifacts ---
print("Loading LightGBM model and artifacts...")
try:
    # Load model, scaler, and feature list together
    model, scaler, features_to_use = joblib.load(MODEL_ARTIFACT_PATH) 
    print(f"LightGBM model, scaler, and {len(features_to_use)} feature columns loaded successfully.")
    # print(f"Features loaded: {features_to_use}") # Optional: Print features
except FileNotFoundError:
    print(f"Error: Artifact file not found: {MODEL_ARTIFACT_PATH}")
    print("Please ensure the walk-forward + final model training script for LightGBM was run and saved the artifacts.")
    exit(1)
except Exception as e:
     print(f"Error loading model/artifacts: {e}")
     exit(1)


# --- Database Connection ---
conn = None
try:
    print(f"Connecting to database '{DB_NAME}'...")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    conn.autocommit = True 
    cursor = conn.cursor()
    print("Connection successful.")
    # Create predictions table (corrected version)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_predictions (
        id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, ticker VARCHAR(10) NOT NULL,
        current_price NUMERIC(12, 4) NOT NULL, predicted_probability NUMERIC(10, 4), 
        trade_signal BOOLEAN, target_price NUMERIC(12, 4), stop_loss NUMERIC(12, 4) );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_predictions_timestamp_ticker ON stock_predictions (timestamp DESC, ticker);")
    print("Prediction table verified/created.")
except psycopg2.Error as e: print(f"DB Setup Error: {e}"); exit(1)
except Exception as e: print(f"Setup Error: {e}"); exit(1)


# --- Enhanced Feature Engineering Function Definition ---
def create_features(df_input): 
    print("Generating features...")
    df = df_input.copy()

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns: {required_cols}")
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception as e:
            print(f"Feature Gen Error converting index: {e}")
            return pd.DataFrame()

    df.sort_index(inplace=True)

    # Existing basic features
    df['return'] = df['close'].pct_change()
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['log_volume'] = np.log1p(df['volume'].replace(0, 1))
    df['price_vwap_diff'] = (df['close'] - df['vwap']).fillna(0)

    # Moving averages & ratios
    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = df['close'].rolling(window=window, min_periods=window).mean()
        df[f'ma_vol_{window}'] = df['volume'].rolling(window=window, min_periods=window).mean()
        df[f'close_ma_{window}_ratio'] = (df['close'] / df[f'ma_{window}']).replace([np.inf, -np.inf], np.nan)
        df[f'volume_ma_{window}_ratio'] = (df['volume'] / df[f'ma_vol_{window}']).replace([np.inf, -np.inf], np.nan)

    # --- New features to add ---

    # Volatility (5-minute rolling standard deviation of returns)
    df['volatility_5m'] = df['return'].rolling(window=5).std()

    # Volume-based features
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_change_1m'] = df['volume'].pct_change()

    # Momentum indicator (RSI)
    if PANDAS_TA_AVAILABLE:
        df['rsi_14'] = df.ta.rsi(length=14)
    else:
        print("Warning: pandas_ta not installed, skipping RSI feature.")
        df['rsi_14'] = np.nan

    # Price range features
    df['high_low_range'] = df['high'] - df['low']
    df['close_high_ratio'] = df['close'] / df['high']
    df['close_low_ratio'] = df['close'] / df['low']

    # Trend indicator (EMA)
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

    # Volume delta
    df['volume_delta'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan)

    # Time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek

    # Drop rows with NaNs resulting from rolling calculations
    df.dropna(inplace=True)

    print(f"Finished feature generation. Shape: {df.shape}")
    return df

# --- Function to Get Recent Data ---
def get_recent_data(ticker='TSLA', limit=FETCH_LIMIT): # Removed minutes argument, just fetch limit
    print(f"Fetching data for {ticker} (limit {limit} rows)...")
    # Fetch last 'limit' points to ensure enough history for feature calculation
    query = """
    SELECT start_time AS time, agg_open AS open, agg_high AS high, agg_low AS low, 
           agg_close AS close, volume, vwap, num_trades      
    FROM stock_aggregates_min
    WHERE symbol = %(ticker)s ORDER BY start_time DESC LIMIT %(limit)s; 
    """
    try:
        df_fetched = pd.read_sql(query, conn, params={'ticker': ticker, 'limit': limit}, index_col='time', parse_dates={'time': {'utc': True}})
        if not df_fetched.empty:
             df_fetched.sort_index(ascending=True, inplace=True) # Sort back to chronological ONLY if data found
        print(f"Returning {len(df_fetched)} data points.")
        return df_fetched
    except Exception as e: print(f"Error fetching recent data: {e}"); return pd.DataFrame() 

# --- Function to Make Predictions ---
def make_prediction(ticker='TSLA'):
    print(f"\n--- Prediction Cycle for {ticker} at {datetime.now(timezone.utc)} ---")
    try:
        # 1. Get recent data 
        # Ensure we fetch enough for the largest lookback window (e.g., 60 for ma_60) + 1
        history_needed = 61 # For 60 period lookback features + current point
        recent_data_raw = get_recent_data(ticker, limit=max(FETCH_LIMIT, history_needed)) # Fetch enough history
        
        if recent_data_raw.empty or len(recent_data_raw) < history_needed: 
            print(f"Not enough recent data points ({len(recent_data_raw)} fetched) < {history_needed} required. Skipping."); return None
            
        # 2. Create features
        feature_df_full = create_features(recent_data_raw) 
        if feature_df_full.empty: print("Feature generation returned empty DataFrame."); return None
        
        # 3. Get the LATEST feature row (MUST correspond to latest time in raw data)
        # Ensure features_to_use columns exist
        missing_feature_cols = [col for col in features_to_use if col not in feature_df_full.columns]
        if missing_feature_cols: print(f"Error: Missing required features after generation: {missing_feature_cols}"); return None
        
        latest_feature_row_df = feature_df_full[features_to_use].iloc[-1:] # Select FEATURES only
        
        if latest_feature_row_df.empty: print("Could not get latest row after feature gen."); return None

        # 4. Check for NaNs in the *required* features for the latest point
        if latest_feature_row_df.isnull().any(axis=1).iloc[0]:
             print("Warning: Latest feature row contains NaNs in required features. Insufficient history for lookbacks?")
             # print(latest_feature_row_df.isnull().sum()) # Uncomment for detailed debug
             return None 

        # 5. Get price/time from the LATEST entry in the ORIGINAL fetched data (index aligns)
        current_time = latest_feature_row_df.index[0] 
        # <<< FIX: Use the raw DataFrame to get the most recent price before feature NaN drops >>>
        # If feature_df_full dropped latest due to NaNs, this would error before
        # Instead, get price from recent_data_raw which we know has the latest timestamp
        current_price = recent_data_raw.loc[current_time, 'close'] 
        # <<< End Fix >>>
        if pd.isna(current_price): print(f"Error: Found NaN for close price at {current_time}"); return None

        print(f"Predicting for time: {current_time}, Price: {current_price:.4f}")

        # 6. Scale the required features for the latest row
        features_scaled = scaler.transform(latest_feature_row_df.values) 

        # 7. Make LightGBM Prediction
        lgbm_prob = 0.0
        try:
            lgbm_prob = model.predict_proba(features_scaled)[0, 1] # Prob of class 1 
            print(f"LGBM Prediction successful (Prob: {lgbm_prob:.4f})")
        except Exception as e: print(f"Error during LightGBM prediction: {e}")

        # 8. Determine Signal and Target/Stop
        final_probability = lgbm_prob 
        trade_signal = final_probability >= PREDICTION_THRESHOLD
        target_price = float(current_price) + TARGET_GAIN  
        stop_loss = float(current_price) - (TARGET_GAIN / 2) 

        # 9. Store prediction
        insert_query = """
        INSERT INTO stock_predictions (timestamp, ticker, current_price, predicted_probability, 
                                       trade_signal, target_price, stop_loss)
        VALUES (%s, %s, %s, %s, %s, %s, %s);"""
        insert_data = ( current_time.to_pydatetime(), ticker, float(current_price), float(final_probability), bool(trade_signal), 
                        float(target_price) if bool(trade_signal) else None, float(stop_loss) if bool(trade_signal) else None )
        try: cursor.execute(insert_query, insert_data); print("Prediction saved to database.")
        except Exception as db_err: print(f"DB Save Error: {db_err}")
        
        # Print summary
        print(f"--- Prediction Summary ---")
        print(f"Time: {current_time}"); print(f"Ticker: {ticker}")
        print(f"Current Price: ${current_price:.4f}")
        print(f"LGBM Probability (Gain>=${TARGET_GAIN:.2f}): {final_probability:.4f}")
        signal_text = 'BUY' if trade_signal else 'HOLD'; print(f"Trade Signal (>{PREDICTION_THRESHOLD:.2f}): {signal_text}")
        if trade_signal: print(f"  Target: ${target_price:.2f}, Stop: ${stop_loss:.2f}")
        
        return insert_data 

    except Exception as e: print(f"FATAL Error in make_prediction: {e}"); import traceback; traceback.print_exc(); return None

# --- Function to run prediction service loop ---
def run_prediction_service(interval_seconds=PREDICTION_INTERVAL_SECONDS):
    print(f"Prediction loop starting. Interval: {interval_seconds} seconds.")
    while True:
        loop_start = time.time()
        try: prediction = make_prediction(TICKER) 
        except Exception as e: print(f"Prediction service loop error: {e}")
        elapsed = time.time() - loop_start
        sleep_time = max(0, interval_seconds - elapsed)
        print(f"Loop finished in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s...")
        time.sleep(sleep_time)

# --- Main thread ---
if __name__ == "__main__": 
     print("Starting prediction service thread...")
     prediction_thread = threading.Thread(target=run_prediction_service, daemon=True); prediction_thread.start()
     print("\nPrediction service running in background."); print("Main thread periodically shows recent predictions (Ctrl+C to exit).")
     try:
         while True:
             time.sleep(30) 
             print("\n--- Recent Stored Predictions ---")
             try:
                 cursor.execute("""SELECT timestamp, ticker, current_price, predicted_probability, trade_signal FROM stock_predictions ORDER BY timestamp DESC LIMIT 5; """)
                 recent_preds = cursor.fetchall()
                 if recent_preds:
                     for pred in recent_preds: ts, tick, price, prob, signal_bool = pred; signal = "BUY" if signal_bool else "HOLD"; print(f"{ts} | {tick} | ${price:.2f} | Prob:{prob:.4f} | {signal}")
                 else: print("No predictions found in database yet.")
             except Exception as e: print(f"Error fetching recent preds: {e}")
     except KeyboardInterrupt: print("\nCtrl+C received. Shutting down...")
     finally:
         if conn: conn.close(); print("Database connection closed.")
         print("Prediction service main thread exiting.")
