# --- Necessary Imports ---
import psycopg2
import pandas as pd
import numpy as np
import time
import os
import math 
from datetime import datetime, timedelta, timezone 
from dotenv import load_dotenv

# ML/DL Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split # Note: We'll use manual slicing for time series
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Plotting and Saving Imports
import matplotlib.pyplot as plt
import joblib 

# Technical Analysis Library Import
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    print("pandas_ta imported successfully.")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas_ta not installed. Skipping TA-Lib based features. (Run: pip install pandas-ta)")

print("\n--- Configuration & Setup ---")

# --- TensorFlow GPU Check ---
print("Checking for GPU...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU Available: Yes, Found {len(physical_devices)} GPU(s)")
    try:
        # Attempt memory growth configuration
        for gpu in physical_devices: # Iterate over all GPUs if more than one
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Could not enable memory growth (may already be initialized): {e}")
else:
    print("GPU Available: No. Training will use CPU.")
print(f"TensorFlow version: {tf.__version__}")

# --- Configuration Loading & Constants ---
load_dotenv() 
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST = "localhost"
DB_PORT = "5433" # Host Port mapped in docker-compose
DB_NAME = "polygondata"
DB_USER = "polygonuser"
TICKER = "TSLA"

# Model & Feature Parameters
SEQ_LENGTH = 60          
LOOKAHEAD_PERIOD = 15 
TARGET_GAIN = 1.0       
TEST_SET_SIZE = 0.2     # Proportion of data for testing (used for slicing index)
VALIDATION_SPLIT = 0.2  # Proportion of training data for validation during fit
EPOCHS = 50             
BATCH_SIZE = 64
LSTM_UNITS_1 = 100 
LSTM_UNITS_2 = 50  
DROPOUT_RATE = 0.2
PREDICTION_THRESHOLD = 0.6 # Backtesting trade signal threshold

# Output filenames
MODEL_SAVE_PATH = f'lstm_model_enhanced_{TICKER.lower()}.keras' 
SCALER_COLS_FILENAME = f'lstm_scaler_columns_enhanced_{TICKER.lower()}.joblib'
HISTORY_PLOT_FILENAME = f'lstm_enhanced_training_history_{TICKER.lower()}.png'
BACKTEST_PLOT_FILENAME = f'lstm_enhanced_backtest_results_{TICKER.lower()}.png'


# --- Database Connection & Data Loading ---
conn = None
df = pd.DataFrame() 

try:
    print(f"\nConnecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    print("Connection successful.")
    print(f"Fetching historical data for {TICKER}...")
    query = """
    SELECT start_time AS time, agg_open AS open, agg_high AS high, agg_low AS low,   
           agg_close AS close, volume, vwap, num_trades      
    FROM stock_aggregates_min WHERE symbol = %(ticker)s AND start_time IS NOT NULL 
    ORDER BY start_time ASC;   
    """
    # Use context manager for connection safety if preferred, but ensure df is accessible outside
    df = pd.read_sql(query, conn, params={'ticker': TICKER}, index_col='time', parse_dates={'time': {'utc': True}})
    print(f"Loaded {len(df)} historical bars for {TICKER} from DB.")

except psycopg2.Error as e: 
    print(f"Database error: {e}")
    exit(1)
except Exception as e: 
    print(f"An error occurred during data loading: {e}")
    exit(1)
finally:
    if conn: 
        conn.close()
        print("Database connection closed.")
        
if df.empty: 
    print("No data loaded. Exiting.")
    exit(1)

# --- Enhanced Feature Engineering Function ---
def create_features(df): 
    print("Generating features...")
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols): 
        print(f"Error: DataFrame missing required columns: {required_cols}"); return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
         try: df.index = pd.to_datetime(df.index, utc=True); print("Converted index to DatetimeIndex.")
         except Exception as e: print(f"Error converting index: {e}"); return pd.DataFrame()
    if not df.index.is_monotonic_increasing: print("Warning: Sorting non-monotonic index..."); df.sort_index(inplace=True)

    df_feat = df.copy()
    print(f"Initial rows for feature gen: {len(df_feat)}")

    # 1. Basic Features
    df_feat['return'] = df_feat['close'].pct_change()
    df_feat['high_low_range'] = df_feat['high'] - df_feat['low']
    df_feat['close_open_diff'] = df_feat['close'] - df_feat['open']
    df_feat['log_volume'] = np.log1p(df_feat['volume'].replace(0, 1)) 
    df_feat['price_vwap_diff'] = df_feat['close'] - df_feat['vwap'] if 'vwap' in df_feat.columns else 0
    df_feat['price_vwap_diff'].fillna(0, inplace=True)

    # 2. Moving Averages & Ratios
    for window in [5, 10, 20, 60]:
        df_feat[f'ma_{window}'] = df_feat['close'].rolling(window=window, min_periods=window).mean()
        df_feat[f'ma_vol_{window}'] = df_feat['volume'].rolling(window=window, min_periods=window).mean()
        # Calculate ratios safely, handle potential division by zero or NaN results
        df_feat[f'close_ma_{window}_ratio'] = (df_feat['close'] / df_feat[f'ma_{window}']).replace([np.inf, -np.inf], np.nan)
        df_feat[f'volume_ma_{window}_ratio'] = (df_feat['volume'] / df_feat[f'ma_vol_{window}']).replace([np.inf, -np.inf], np.nan)

    # 3. Momentum / ROC
    for window in [1, 5, 10, 20]:
        df_feat[f'roc_{window}'] = df_feat['close'].pct_change(periods=window)

    # 4. Volatility
    df_feat['volatility_20'] = df_feat['return'].rolling(window=20, min_periods=20).std()
    
    if PANDAS_TA_AVAILABLE:
        print("Calculating pandas_ta features (ATR, Bollinger, RSI)...")
        try:
            df_feat.ta.atr(length=14, append=True) # 'ATRr_14'
            df_feat.ta.bbands(length=20, std=2, append=True) # Adds 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'
            df_feat.ta.rsi(length=14, append=True) # 'RSI_14'
            print("TA features added.")
        except Exception as e:
            print(f"Warning: Error calculating pandas_ta features: {e}")
            # Add placeholder columns if calculation fails, to prevent dropna errors later
            if 'ATRr_14' not in df_feat.columns: df_feat['ATRr_14'] = np.nan
            if 'BBB_20_2.0' not in df_feat.columns: df_feat['BBB_20_2.0'] = np.nan # Bandwidth
            if 'BBP_20_2.0' not in df_feat.columns: df_feat['BBP_20_2.0'] = np.nan # Percent B
            # Add placeholders for other bbands cols if needed by feature list later
            if 'BBL_20_2.0' not in df_feat.columns: df_feat['BBL_20_2.0'] = np.nan
            if 'BBM_20_2.0' not in df_feat.columns: df_feat['BBM_20_2.0'] = np.nan
            if 'BBU_20_2.0' not in df_feat.columns: df_feat['BBU_20_2.0'] = np.nan
            if 'RSI_14' not in df_feat.columns: df_feat['RSI_14'] = np.nan
    else: 
        df_feat['ATRr_14'] = np.nan
        df_feat['BBB_20_2.0'] = np.nan; df_feat['BBP_20_2.0'] = np.nan
        df_feat['BBL_20_2.0'] = np.nan; df_feat['BBM_20_2.0'] = np.nan; df_feat['BBU_20_2.0'] = np.nan
        df_feat['RSI_14'] = np.nan

    # 5. Volume-based features (Relative)
    df_feat['volume_delta'] = df_feat['volume'].pct_change().replace([np.inf, -np.inf], np.nan)

    # 6. Time-based features
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour']/24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour']/24.0)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['dayofweek']/7.0)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['dayofweek']/7.0)

    # Define the FINAL feature set to check for NaNs before dropping
    final_feature_set = [col for col in df_feat.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']]

    # Handle NaNs created by rolling windows etc.
    initial_len = len(df_feat)
    df_feat.dropna(subset=final_feature_set, inplace=True) 
    final_len = len(df_feat)
    print(f"Rows after dropping initial NaNs based on features: {final_len} (dropped {initial_len-final_len})")
    
    if final_len == 0:
         print("Error: All rows dropped after feature calculation NaNs.")
         return pd.DataFrame()

    print(f"Finished generating {len(final_feature_set)} features. Final shape before target: {df_feat.shape}")
    return df_feat


# --- Target Definition Function ---
def add_target(df, lookahead=LOOKAHEAD_PERIOD, gain=TARGET_GAIN):
    print(f"Adding target variable (lookahead={lookahead} mins, gain=${gain})...")
    if df.empty: return df
    df_target = df.copy()
    if 'close' not in df_target.columns: return pd.DataFrame() 

    df_target['future_price'] = df_target['close'].shift(-lookahead)
    df_target['target'] = ((df_target['future_price'] - df_target['close']) >= gain).astype(int)
    
    initial_len = len(df_target)
    df_target.dropna(subset=['future_price', 'target'], inplace=True)
    final_len = len(df_target)
    print(f"Rows after dropping rows without future target: {final_len} (removed {initial_len - final_len})")
    return df_target

# --- Backtesting Function Definition ---
def backtest_lstm_model(df_features, keras_model, fitted_scaler, feature_cols, sequence_length=SEQ_LENGTH, lookahead=LOOKAHEAD_PERIOD, gain_threshold=TARGET_GAIN, prob_threshold=PREDICTION_THRESHOLD):
    print("Scaling full feature set for backtest sequence creation...")
    
    missing_cols = [col for col in feature_cols if col not in df_features.columns]
    if missing_cols: print(f"Error in backtest: Missing feature columns in df_features: {missing_cols}"); return pd.DataFrame() 

    # Scale only the necessary feature columns    
    X_full_scaled = fitted_scaler.transform(df_features[feature_cols])
    # Create a temporary DataFrame with scaled data and original index for easy lookup
    X_full_scaled_df = pd.DataFrame(X_full_scaled, index=df_features.index, columns=feature_cols)
    
    results = []
    # Ensure iteration indices are valid based on actual length AFTER target drop might occur
    # We need data up to index i-1 for sequence, and outcome up to index i+lookahead
    # The df_features passed in should ALREADY have target and future price (and NaNs from that dropped)
    print(f"Iterating for backtest (Range: {sequence_length} to {len(df_features)} - prediction determines outcome {lookahead} steps later)")
    
    num_predictions = 0
    for i in range(sequence_length, len(df_features)): # Iterate up to the end, prediction looks ahead
        
        # Sequence is features from i-sequence_length to i-1
        start_idx_features = i - sequence_length
        end_idx_features = i 
        
        # Ensure indices are valid (especially near the start)
        if start_idx_features < 0: continue # Should not happen if loop starts at seq_length
        
        current_seq_scaled = X_full_scaled[start_idx_features:end_idx_features]

        if current_seq_scaled.shape[0] != sequence_length: continue 

        current_seq_input = np.expand_dims(current_seq_scaled, axis=0)
        
        # --- Make Prediction ---
        prob = 0.0
        try:
            prob = keras_model.predict(current_seq_input, verbose=0)[0][0]
            num_predictions += 1
        except Exception as e:
            print(f"Error predicting at step index {i}: {e}")
            continue # Skip step on prediction error
            
        if i % 500 == 0: print(f"  Backtest progress: Predicted step index {i}...")

        # --- Get relevant prices for analysis ---
        # Decision/Current Price is at index i-1 (the end of the sequence)
        decision_time_index = i - 1 
        decision_timestamp = df_features.index[decision_time_index]
        current_price_at_decision = df_features.iloc[decision_time_index]['close']
        
        # Outcome price is lookahead periods *after* the decision price
        outcome_time_index = decision_time_index + lookahead
        # Check if outcome index is within bounds
        if outcome_time_index >= len(df_features): continue # Cannot evaluate outcome for sequences near the end

        future_price_at_outcome = df_features.iloc[outcome_time_index]['close']
        actual_gain = future_price_at_outcome - current_price_at_decision
        actual_outcome_met_target = actual_gain >= gain_threshold

        # --- Signal and Result Logging ---
        would_trade = prob >= prob_threshold
        prediction_correct = (would_trade == actual_outcome_met_target)

        results.append({
            'timestamp': decision_timestamp,         # Time of decision
            'current_price': current_price_at_decision, # Price when decision made
            'future_price': future_price_at_outcome,    # Actual outcome price
            'probability': prob,
            'actual_gain': actual_gain,
            'would_trade': would_trade,             # Based on threshold
            'actual_outcome': actual_outcome_met_target, # Did actual meet target?
            'prediction_correct': prediction_correct   # Did prediction match actual outcome?
        })
            
    print(f"Finished backtest loop, {num_predictions} predictions made for evaluation.")
    return pd.DataFrame(results)


# ==============================================================================
#                             MAIN EXECUTION FLOW
# ==============================================================================

# 1. Create Features
features_only_df = create_features(df)
if features_only_df.empty: print("Feature generation failed. Exiting."); exit(1)

# 2. Add Target Variable
feature_df = add_target(features_only_df) # Use the result from create_features
if feature_df.empty: print("Target generation failed or resulted in empty dataframe. Exiting."); exit(1)

print(f"\nFinal DataFrame shape for modeling: {feature_df.shape}")
print(f"Target value counts:\n{feature_df['target'].value_counts(normalize=True)}")


# --- Data Preparation for Model ---
print("\n--- Data Preparation for Model ---")
# Define the final list of features to be used by the model
# Excludes raw OHLCV, helpers like hour/dayofweek, and target/future price
features_to_use = [col for col in feature_df.columns if col not in 
    ['target', 'future_price', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']]
print(f"Selected {len(features_to_use)} features for model input: {features_to_use}")

X = feature_df[features_to_use].values 
y = feature_df['target'].values

# --- Train/Test Split (Time Series) ---
print(f"Splitting data (Test size: {TEST_SET_SIZE})...")
split_index = int(len(X) * (1 - TEST_SET_SIZE))
X_train_raw, X_test_raw = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:] 
print(f"Raw Train shape: {X_train_raw.shape}, Raw Test shape: {X_test_raw.shape}")

# --- Scaling ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
print("Scaling complete.")

# --- Sequence Creation ---
print(f"Creating sequences (Sequence Length: {SEQ_LENGTH})...")
def create_sequences_for_lstm(features, targets, seq_length):
    X_seq, y_seq = [], []
    # The loop should go from seq_length up to the length of the FEATURE set
    # to ensure we have a corresponding target available at index i
    for i in range(seq_length - 1, len(features) - 1): 
        start_idx = i - (seq_length - 1)
        end_idx = i + 1 # Include index i 
        X_seq.append(features[start_idx:end_idx])
        # Target corresponds to the step *immediately after* the sequence ends (index i+1)
        # but since the original y was sliced the same as X_raw, y[i] actually 
        # corresponds to the target AFTER the sequence features[start_idx:end_idx]
        # Let's rethink: target needs to align with the *end* of the sequence prediction point
        # Target y[i] should correspond to sequence ending at features[i-1]
        target_idx = i # Use target at index i 
        if target_idx < len(targets): # Ensure target index is valid
            y_seq.append(targets[target_idx])
        else:
             # This case should not happen if len(features) > seq_length
             # Add a check or break? For now assume alignment from split.
             pass 
    # Need to make sure X and y have the same number of samples after loop
    X_seq_arr = np.array(X_seq)
    y_seq_arr = np.array(y_seq)
    
    # Trim potential mismatch if loop logic slightly off (should investigate if happens)
    min_len = min(len(X_seq_arr), len(y_seq_arr))
    return X_seq_arr[:min_len], y_seq_arr[:min_len]


X_train_seq, y_train_seq_final = create_sequences_for_lstm(X_train_scaled, y_train, SEQ_LENGTH)
X_test_seq, y_test_seq_final = create_sequences_for_lstm(X_test_scaled, y_test, SEQ_LENGTH)

print(f"Train sequences shape: {X_train_seq.shape}, Train target shape: {y_train_seq_final.shape}")
print(f"Test sequences shape: {X_test_seq.shape}, Test target shape: {y_test_seq_final.shape}")
if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0: print("Sequence creation failed."); exit(1)

# --- LSTM Model ---
print("\n--- Building LSTM Model ---")
n_features = X_train_seq.shape[2] 
print(f"Number of features input to LSTM: {n_features}")
model = Sequential([
    LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=(SEQ_LENGTH, n_features)),
    Dropout(DROPOUT_RATE),
    LSTM(LSTM_UNITS_2, return_sequences=False), 
    Dropout(DROPOUT_RATE),
    Dense(25, activation='relu'), 
    Dense(1, activation='sigmoid') 
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

# --- Class Weights ---
unique_classes, counts = np.unique(y_train_seq_final, return_counts=True)
print(f"Target distribution in training sequences: {dict(zip(unique_classes, counts))}")
class_weight_dict = None
if len(unique_classes) == 2: 
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_seq_final)
    class_weight_dict = dict(enumerate(class_weights)) 
    print(f"Using Class Weights: {class_weight_dict}")
else: print("Warning: Training data has only one class or is empty.")

# --- Train Model ---
if class_weight_dict is None and len(unique_classes) < 2:
     print("Cannot train model with only one class in target variable. Exiting.")
     exit(1)

print("\n--- Training LSTM Model with Enhanced Features ---")
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
history = model.fit( X_train_seq, y_train_seq_final, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, class_weight=class_weight_dict, callbacks=[early_stopping, reduce_lr], verbose=1)

# --- Evaluate Model ---
print("\n--- Evaluating Enhanced LSTM Model ---")
results = model.evaluate(X_test_seq, y_test_seq_final, verbose=0)
metric_names = model.metrics_names
print("Test Set Performance:")
for name, value in zip(metric_names, results): print(f"  {name}: {value:.4f}")

# --- Predictions & Report ---
print("\nGenerating classification report...")
y_pred_proba = model.predict(X_test_seq)
y_pred_binary = (y_pred_proba > PREDICTION_THRESHOLD).astype(int).flatten()
print(f"\nEnhanced LSTM Model Performance (Test Set, Threshold={PREDICTION_THRESHOLD}):")
print(classification_report(y_test_seq_final.flatten(), y_pred_binary, zero_division=0))

# --- Save Artifacts ---
print("\n--- Saving Artifacts (Enhanced Model) ---")
try:
    model.save(MODEL_SAVE_PATH); print(f"Enhanced LSTM model saved to {MODEL_SAVE_PATH}")
    joblib.dump((scaler, features_to_use), SCALER_COLS_FILENAME); print(f"Scaler and features saved to {SCALER_COLS_FILENAME}")
except Exception as e: print(f"Error saving artifacts: {e}")

# --- Plot History ---
print("\n--- Plotting Training History ---")
try:
    plt.figure(figsize=(12, 6)); plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss'); plt.title('Model Loss'); plt.ylabel('Loss')
    plt.xlabel('Epoch'); plt.legend(); plt.subplot(1, 2, 2); plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if 'precision' in history.history: plt.plot(history.history['val_precision'], label='Val Precision')
    if 'recall' in history.history: plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Model Accuracy / Metrics'); plt.ylabel('Metric Value'); plt.xlabel('Epoch'); plt.legend()
    plt.tight_layout(); plt.savefig(HISTORY_PLOT_FILENAME); plt.close()
    print(f"Training history saved to {HISTORY_PLOT_FILENAME}")
except Exception as e: print(f"Error plotting history: {e}")

# --- Run Backtest ---
print("\n--- Running Basic LSTM Backtest (Enhanced Model) ---")
if not feature_df.empty:
    lstm_backtest_results = backtest_lstm_model(
        df_features=feature_df, keras_model=model, fitted_scaler=scaler, 
        feature_cols=features_to_use, sequence_length=SEQ_LENGTH, 
        lookahead=LOOKAHEAD_PERIOD, gain_threshold=TARGET_GAIN, 
        prob_threshold=PREDICTION_THRESHOLD
    )
    # --- Analyze Backtest Results ---
    print("\n--- LSTM Backtest Analysis (Enhanced Model) ---")
    if not lstm_backtest_results.empty:
        print(f"Total backtest prediction points: {len(lstm_backtest_results)}")
        lstm_trades = lstm_backtest_results[lstm_backtest_results['would_trade']].copy()
        print(f"'Trade' signals generated (prob >= {PREDICTION_THRESHOLD}): {len(lstm_trades)} ({len(lstm_trades)/len(lstm_backtest_results)*100:.2f}%)")
        if not lstm_trades.empty:
            winning_lstm_trades = lstm_trades[lstm_trades['actual_outcome'] == True] 
            print(f"Successful 'Trades' (Signal matched ${TARGET_GAIN:.2f}+ gain): {len(winning_lstm_trades)} ({(len(winning_lstm_trades)/len(lstm_trades)*100) if len(lstm_trades)>0 else 0:.2f}%)")
            print(f"Average actual gain/loss on signaled 'trades': ${lstm_trades['actual_gain'].mean():.2f}")
            print(f"Average actual gain on WINNING signaled 'trades': ${winning_lstm_trades['actual_gain'].mean() if not winning_lstm_trades.empty else 0:.2f}")

            # --- Visualize Backtest ---
            print("\n--- Visualizing LSTM Backtest (Enhanced Model) ---")
            try:
                plt.figure(figsize=(15, 8)); plt.plot(feature_df.index, feature_df['close'], 'k-', alpha=0.3, label=f'{TICKER} Price')
                if not winning_lstm_trades.empty: plt.scatter(winning_lstm_trades['timestamp'], winning_lstm_trades['current_price'], color='lime', marker='^', s=50, label='Successful Signal')
                losing_lstm_trades = lstm_trades[lstm_trades['actual_outcome'] == False]
                if not losing_lstm_trades.empty: plt.scatter(losing_lstm_trades['timestamp'], losing_lstm_trades['current_price'], color='red', marker='v', s=50, label='Unsuccessful Signal')
                plt.title(f'Enhanced LSTM Backtest ({TICKER}): Signals (Prob>={PREDICTION_THRESHOLD}) & Outcomes'); plt.legend(); plt.ylabel("Price ($)"); plt.xlabel("Time"); plt.savefig(BACKTEST_PLOT_FILENAME); plt.close()
                print(f"Saved Enhanced LSTM backtest visualization to {BACKTEST_PLOT_FILENAME}")
            except Exception as e: print(f"Error plotting backtest results: {e}")
        else: print("No 'trade' signals generated during backtest.")
    else: print("LSTM Backtest produced no results.")
else: print("Skipping backtest as feature_df was empty.")

print("\nScript Finished.")
