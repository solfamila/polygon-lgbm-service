import psycopg2
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import joblib # Using joblib for scaler, as it's standard sklearn

print("--- Configuration & Setup ---")

# --- TensorFlow GPU Check ---
print("Checking for GPU...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU Available: Yes, Found {len(physical_devices)} GPU(s)")
    try:
        # Enable memory growth for the first GPU to avoid allocating all memory at once
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth enabled for GPU 0.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Could not enable memory growth: {e}")
else:
    print("GPU Available: No")
print(f"TensorFlow version: {tf.__version__}")

# --- Configuration Loading ---
load_dotenv() 
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "polygondata"
DB_USER = "polygonuser"
TICKER = "TSLA"
# LSTM Hyperparameters & Settings
SEQ_LENGTH = 60          # Sequence length (e.g., look back 60 minutes)
LOOKAHEAD_PERIOD = 15 # How far ahead to predict (e.g., 15 minutes)
TARGET_GAIN = 1.0       # Target gain in dollars
TEST_SET_SIZE = 0.2     # Proportion of data for testing
VALIDATION_SPLIT = 0.2  # Proportion of *training* data for validation during fit
EPOCHS = 50             # Max epochs (EarlyStopping will likely stop it sooner)
BATCH_SIZE = 64
LSTM_UNITS_1 = 100
LSTM_UNITS_2 = 50
DROPOUT_RATE = 0.2
PREDICTION_THRESHOLD = 0.6 # Probability threshold to trigger a 'buy' signal in backtest

# --- Database Connection & Data Loading ---
conn = None
df = pd.DataFrame() # Initialize empty DataFrame

try:
    print(f"\nConnecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    print("Connection successful.")

    print(f"Fetching historical data for {TICKER}...")
    query = """
    SELECT 
        start_time AS time, 
        agg_open   AS open,  
        agg_high   AS high,  
        agg_low    AS low,   
        agg_close  AS close, 
        volume,
        vwap,            -- Added VWAP
        num_trades       -- Added num_trades
    FROM 
        stock_aggregates_min 
    WHERE 
        symbol = %(ticker)s 
        AND start_time IS NOT NULL -- Ensure timestamp isn't null
    ORDER BY 
        start_time ASC;   
    """
    df = pd.read_sql(query, conn, params={'ticker': TICKER}, index_col='time', parse_dates={'time': {'utc': True}})
    # Ensure index is timezone-aware (UTC is usually good practice)
    # df.index = df.index.tz_localize('UTC') # If not already timezone-aware from DB

    print(f"Loaded {len(df)} historical bars for {TICKER} from DB.")

except psycopg2.Error as e:
    print(f"Database error: {e}")
    # Optionally try loading from CSV as fallback
    # try: 
    #    print("Database failed, attempting to load from CSV...")
    #    df = pd.read_csv(f"{TICKER}_minute_2025-04-12.csv", index_col='timestamp', parse_dates=True) # Assuming filename pattern
    #    print(f"Loaded {len(df)} rows from CSV.")
    # except Exception as csv_e:
    #    print(f"CSV loading failed: {csv_e}")
    #    exit(1)
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

# --- Feature Engineering ---
print("\n--- Feature Engineering ---")
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
    df['future_price'] = df['close'].shift(-lookahead)
    df['target'] = ((df['future_price'] - df['close']) >= gain).astype(int)
    
    # Drop rows with NaNs created by feature engineering
    print(f"Rows before dropna: {len(df)}")
    df.dropna(inplace=True)
    print(f"Rows after dropna: {len(df)}")
    
    return df

feature_df = create_features(df)

if feature_df.empty:
    print("Feature DataFrame is empty after dropping NaNs. Exiting.")
    exit(1)

print(f"\nFeature Columns: {feature_df.drop(columns=['target', 'future_price']).columns.tolist()}")
print(f"Number of $1 gain opportunities: {feature_df['target'].sum()} ({feature_df['target'].mean()*100:.2f}%)")

# --- Data Preparation for LSTM ---
print("\n--- LSTM Data Preparation ---")

# Select features (X) and target (y)
# IMPORTANT: Exclude future_price and target from features! Also raw OHLCV often excluded if derived features are used.
features_to_use = [col for col in feature_df.columns if col not in ['target', 'future_price', 'open', 'high', 'low', 'close', 'volume']] 
print(f"Using features: {features_to_use}")
X_data = feature_df[features_to_use].values
y_data = feature_df['target'].values

# --- Splitting MUST be done BEFORE scaling to prevent data leakage ---
print(f"Splitting data (Test size: {TEST_SET_SIZE})...")
# Important: shuffle=False for time series data!
split_index = int(len(X_data) * (1 - TEST_SET_SIZE))
X_train_raw, X_test_raw = X_data[:split_index], X_data[split_index:]
y_train_seq, y_test_seq = y_data[:split_index], y_data[split_index:] # Target doesn't need scaling/sequencing before split

print(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")
if X_train_raw.shape[0] < SEQ_LENGTH or X_test_raw.shape[0] < SEQ_LENGTH:
     print(f"Error: Not enough data in train/test sets to create sequences of length {SEQ_LENGTH}. Try smaller SEQ_LENGTH or less test data.")
     exit(1)


# --- Scale Features (Fit ONLY on Training Data) ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# --- Create Sequences ---
print(f"Creating sequences (Sequence Length: {SEQ_LENGTH})...")
def create_sequences(data, target, seq_length):
    X, y = [], []
     # Start loop only if enough data exists AFTER the sequence length
    for i in range(seq_length, len(data)): 
        X.append(data[i-seq_length : i]) # Sequence ends *before* current time i
        y.append(target[i])              # Target is at current time i
    return np.array(X), np.array(y)

# Create sequences for train and test sets separately
X_train_seq, y_train_seq_final = create_sequences(X_train_scaled, y_train_seq, SEQ_LENGTH)
X_test_seq, y_test_seq_final = create_sequences(X_test_scaled, y_test_seq, SEQ_LENGTH)

print(f"Train sequences shape: {X_train_seq.shape}, Train target shape: {y_train_seq_final.shape}")
print(f"Test sequences shape: {X_test_seq.shape}, Test target shape: {y_test_seq_final.shape}")

if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
     print(f"Error: Sequence creation resulted in zero samples. Check data lengths and SEQ_LENGTH.")
     exit(1)

# --- LSTM Model Definition ---
print("\n--- Building LSTM Model ---")
n_features = X_train_seq.shape[2] # Get number of features from sequence shape
print(f"Number of features input to LSTM: {n_features}")

model = Sequential([
    LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=(SEQ_LENGTH, n_features)),
    Dropout(DROPOUT_RATE),
    LSTM(LSTM_UNITS_2, return_sequences=False), # Last LSTM layer returns a single vector
    Dropout(DROPOUT_RATE),
    Dense(25, activation='relu'), # Optional intermediate dense layer
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

model.summary()

# --- Compile Model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Standard optimizer
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')] # Add Precision/Recall
)

# --- Class Weights ---
# Calculate based on the *final* sequenced training targets
unique_classes, counts = np.unique(y_train_seq_final, return_counts=True)
print(f"Target distribution in training sequences: {dict(zip(unique_classes, counts))}")

if len(unique_classes) == 2: # Proceed only if both classes are present
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train_seq_final
    )
    class_weight_dict = {unique_classes[i]: class_weights[i] for i in range(len(unique_classes))}
    print(f"Using Class Weights: {class_weight_dict}")
else:
    print("Warning: Training data contains only one class. Cannot use balanced weights.")
    class_weight_dict = None


# --- Train Model ---
print("\n--- Training LSTM Model ---")
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss',      # Monitor validation loss
                             patience=5,           # Stop after 5 epochs with no improvement
                             restore_best_weights=True, # Keep the best model found
                             verbose=1) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=3, 
                              min_lr=0.00001,
                              verbose=1)

history = model.fit(
    X_train_seq, y_train_seq_final,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT, # Use part of training seq for validation
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1 # Show progress bar
)

# --- Evaluate Model ---
print("\n--- Evaluating LSTM Model ---")
# Evaluate on the test sequences
results = model.evaluate(X_test_seq, y_test_seq_final, verbose=0)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")
if len(results) > 2: print(f"Test Precision: {results[2]:.4f}")
if len(results) > 3: print(f"Test Recall: {results[3]:.4f}")


# --- Make Predictions & Detailed Report ---
print("\nGenerating classification report...")
y_pred_proba = model.predict(X_test_seq)
y_pred_binary = (y_pred_proba > PREDICTION_THRESHOLD).astype(int).flatten() # Apply threshold for binary prediction

print("\nLSTM Model Performance (Test Set):")
# Make sure y_test_seq_final is flattened if it isn't already
print(classification_report(y_test_seq_final.flatten(), y_pred_binary, zero_division=0))

# --- Save Model, Scaler, and Columns ---
print("\n--- Saving Artifacts ---")
# Save Keras model
model_save_path = 'lstm_model_tsla.keras' 
try:
    model.save(model_save_path)
    print(f"LSTM model saved to {model_save_path}/")
except Exception as e:
     print(f"Error saving Keras model: {e}")
     
# Save the scaler and feature columns used (crucial for prediction/backtesting)
artifacts_filename = 'lstm_scaler_columns_tsla.joblib'
try:
    joblib.dump((scaler, features_to_use), artifacts_filename)
    print(f"Scaler and feature columns saved to {artifacts_filename}")
except Exception as e:
     print(f"Error saving scaler/columns with joblib: {e}")

# --- Plot Training History ---
print("\n--- Plotting Training History ---")
try:
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # If using Precision/Recall metrics:
    if 'precision' in history.history: plt.plot(history.history['val_precision'], label='Val Precision')
    if 'recall' in history.history: plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Model Accuracy / Metrics')
    plt.ylabel('Metric Value')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    hist_filename = 'lstm_training_history.png'
    plt.savefig(hist_filename)
    plt.close()
    print(f"Training history saved to {hist_filename}")
except Exception as e:
    print(f"Error plotting history: {e}")


# --- Basic LSTM Backtesting ---
print("\n--- Running Basic LSTM Backtest ---")

def backtest_lstm_model(df_features, keras_model, fitted_scaler, feature_cols, sequence_length=SEQ_LENGTH, lookahead=LOOKAHEAD_PERIOD, gain_threshold=TARGET_GAIN, prob_threshold=PREDICTION_THRESHOLD):
    print("Scaling full feature set for backtest sequence creation...")
    # Scale the entire feature set using the scaler FIT ON TRAINING DATA
    # Select only the columns the model was trained on
    X_full_scaled = fitted_scaler.transform(df_features[feature_cols])
    
    results = []
    print(f"Iterating through data for backtest (start index: {sequence_length}, end index: {len(df_features) - lookahead})...")
    
    # Iterate through the TIMESTAMPS of the original feature_df
    # We predict based on data *up to* time t-1 to decide action *at* time t
    for i in range(sequence_length, len(df_features) - lookahead):
        
        # Sequence ends at index i-1 (data available before making decision at time i)
        start_idx = i - sequence_length
        end_idx = i
        current_seq_scaled = X_full_scaled[start_idx:end_idx]
        
        # Check sequence shape (should be seq_length x n_features)
        if current_seq_scaled.shape[0] != sequence_length:
            print(f"Skipping index {i}, invalid sequence shape {current_seq_scaled.shape}")
            continue 
            
        # Add batch dimension for LSTM input
        current_seq_input = np.expand_dims(current_seq_scaled, axis=0)
        
        # Predict probability for the NEXT step (based on sequence ending at t-1)
        try:
            prob = keras_model.predict(current_seq_input, verbose=0)[0][0]
        except Exception as e:
            print(f"Error predicting at index {i}: {e}")
            prob = 0.0 # Assign neutral probability on error

        # Add intermittent progress printing
        if i % 500 == 0: # Print every 500 steps
            print(f"  Backtest progress: Processed step {i}/{len(df_features) - lookahead}...")

        # --- Get actual outcomes based on index i and lookahead ---
        current_index_time = df_features.index[i] # Timestamp for decision/current price
        lookahead_index_time = df_features.index[i + lookahead] # Timestamp of future price

        current_price = df_features.loc[current_index_time, 'close']
        future_price = df_features.loc[lookahead_index_time, 'close']
        actual_gain = future_price - current_price
        
        # Decide trade based on threshold
        would_trade = prob >= prob_threshold
        
        # Determine if the prediction was "correct" relative to the defined target
        actual_outcome_met_target = actual_gain >= gain_threshold
        prediction_correct = (would_trade == actual_outcome_met_target)

        # Record results
        results.append({
            'timestamp': current_index_time, # Time of decision/entry price
            'current_price': current_price,
            'future_price': future_price,   # Actual price 'lookahead' periods later
            'probability': prob,
            'actual_gain': actual_gain,
            'would_trade': would_trade,
            'actual_outcome': actual_outcome_met_target, # Did the actual price meet the gain target?
            'prediction_correct': prediction_correct # Did prediction match actual outcome?
        })
        
        # Optional: Print progress intermittently
        # if i % 1000 == 0:
        #    print(f"Backtest progress: processed index {i}")
            
    if not results:
        print("Warning: No results generated during backtest.")
        return pd.DataFrame()
        
    print(f"Finished backtest loop, {len(results)} predictions made.")
    return pd.DataFrame(results)

# Run LSTM backtest using the feature_df (which has NaNs dropped)
if not feature_df.empty:
    lstm_backtest_results = backtest_lstm_model(
        df_features=feature_df,       # The dataframe *after* feature engineering and dropna
        keras_model=model,            # The trained Keras model
        fitted_scaler=scaler,         # The scaler fitted on training data
        feature_cols=features_to_use  # The list of columns used for training
    )

    # --- Analyze LSTM Backtest Results ---
    print("\n--- LSTM Backtest Analysis ---")
    if not lstm_backtest_results.empty:
        print(f"Total backtest prediction points: {len(lstm_backtest_results)}")
        
        lstm_trades = lstm_backtest_results[lstm_backtest_results['would_trade']].copy()
        print(f"'Trade' signals generated (prob >= {PREDICTION_THRESHOLD}): {len(lstm_trades)} ({len(lstm_trades)/len(lstm_backtest_results)*100:.2f}%)")

        if not lstm_trades.empty:
            # Win rate based on if the signal matched the actual outcome (price >= $1 gain)
            winning_lstm_trades = lstm_trades[lstm_trades['actual_outcome'] == True] # Where trade signal aligned with actual $1+ gain
            
            print(f"Successful 'Trades' (Signal matched $1+ gain): {len(winning_lstm_trades)} ({len(winning_lstm_trades)/len(lstm_trades)*100:.2f}%)")
            print(f"Average actual gain/loss on signaled 'trades': ${lstm_trades['actual_gain'].mean():.2f}")
            print(f"Average actual gain on WINNING signaled 'trades': ${winning_lstm_trades['actual_gain'].mean() if not winning_lstm_trades.empty else 0:.2f}")

            # --- Visualize LSTM Backtest ---
            print("\n--- Visualizing LSTM Backtest ---")
            try:
                plt.figure(figsize=(15, 8))
                # Plot original close price using the full feature_df index
                plt.plot(feature_df.index, feature_df['close'], 'k-', alpha=0.3, label=f'{TICKER} Price')
                
                # Plot trade signals - use current_price at the time of the signal
                if not winning_lstm_trades.empty:
                    plt.scatter(winning_lstm_trades['timestamp'], 
                                winning_lstm_trades['current_price'],
                                color='lime', marker='^', s=50, label='Successful Signal') # Use brighter green, upward triangle
                                
                losing_lstm_trades = lstm_trades[lstm_trades['actual_outcome'] == False]
                if not losing_lstm_trades.empty:
                     plt.scatter(losing_lstm_trades['timestamp'], 
                                losing_lstm_trades['current_price'],
                                color='red', marker='v', s=50, label='Unsuccessful Signal') # Downward triangle

                plt.title(f'LSTM Backtest Results ({TICKER}): Buy Signals (Prob>={PREDICTION_THRESHOLD}) and Actual Outcomes')
                plt.legend()
                plt.ylabel("Price ($)")
                plt.xlabel("Time")
                backtest_filename = 'lstm_backtest_results.png'
                plt.savefig(backtest_filename)
                plt.close()
                print(f"Saved LSTM backtest visualization to {backtest_filename}")
            except Exception as e:
                print(f"Error plotting backtest results: {e}")

        else:
             print("No 'trade' signals were generated based on the threshold.")
    else:
        print("LSTM Backtest produced no results.")
else:
    print("Skipping LSTM backtest as feature_df was empty.")

print("\nScript Finished.")
