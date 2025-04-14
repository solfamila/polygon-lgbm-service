# --- Necessary Imports ---
import psycopg2
import pandas as pd
import numpy as np
import time
import os
import math 
from datetime import datetime, timedelta, timezone 
from dotenv import load_dotenv

# --- NEW: XGBoost Import ---
import xgboost as xgb 

# ML/DL & Utility Imports (keep relevant ones)
from sklearn.model_selection import train_test_split # Using manual split, but keep for potential later use
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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

# --- Suppress specific known Pandas FutureWarning ---
import warnings
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.*")


print("\n--- Configuration & Setup ---")

# --- Configuration Loading & Constants ---
load_dotenv() 
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST = "localhost"
DB_PORT = "5433" # Host Port mapped in docker-compose
DB_NAME = "polygondata"
DB_USER = "polygonuser"
TICKER = "TSLA"

# Parameters 
LOOKAHEAD_PERIOD = 15 
TARGET_GAIN = 1.0       
TEST_SET_SIZE = 0.2     
PREDICTION_THRESHOLD = 0.6 # Backtesting trade signal threshold
EARLY_STOPPING_ROUNDS_XGB = 20 # Define early stopping rounds for XGBoost

# Define features to exclude from model input
EXCLUDE_FROM_FEATURES = ['target', 'future_price', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']

# Output filenames
MODEL_SAVE_PATH = f'xgboost_model_enhanced_{TICKER.lower()}.joblib' 
SCALER_COLS_FILENAME = f'xgboost_scaler_columns_enhanced_{TICKER.lower()}.joblib'
CONFUSION_MATRIX_FILENAME = f'xgboost_confusion_matrix_{TICKER.lower()}.png'
FEATURE_IMPORTANCE_FILENAME = f'xgboost_feature_importance_{TICKER.lower()}.png'
BACKTEST_PLOT_FILENAME = f'xgboost_enhanced_backtest_results_{TICKER.lower()}.png'


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
    df_feat['price_vwap_diff'] = df_feat['price_vwap_diff'].fillna(0) 


    # 2. Moving Averages & Ratios
    for window in [5, 10, 20, 60]:
        df_feat[f'ma_{window}'] = df_feat['close'].rolling(window=window, min_periods=window).mean()
        df_feat[f'ma_vol_{window}'] = df_feat['volume'].rolling(window=window, min_periods=window).mean()
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
            required_ta_cols = ['high', 'low', 'close', 'volume'] 
            if not all(col in df_feat.columns for col in required_ta_cols):
                 raise ValueError(f"Missing columns for pandas_ta: Needs {required_ta_cols}")
            
            df_feat.ta.atr(length=14, append=True) 
            bbands_df = df_feat.ta.bbands(length=20, std=2, append=False) 
            if bbands_df is not None and not bbands_df.empty:
                df_feat = pd.concat([df_feat, bbands_df], axis=1) 
            df_feat.ta.rsi(length=14, append=True)
            print("TA features added.")
            
        except Exception as e:
            print(f"Warning: Error calculating pandas_ta features: {e}")
            # Define placeholders if calculation fails
            ta_cols_to_add = ['ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'RSI_14']
            for col in ta_cols_to_add:
                 if col not in df_feat.columns: df_feat[col] = np.nan
    else: 
        ta_cols_to_add = ['ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'RSI_14']
        for col in ta_cols_to_add:
             if col not in df_feat.columns: df_feat[col] = np.nan

    # 5. Volume-based features (Relative)
    df_feat['volume_delta'] = df_feat['volume'].pct_change().replace([np.inf, -np.inf], np.nan)

    # 6. Time-based features
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour']/24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour']/24.0)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['dayofweek']/7.0)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['dayofweek']/7.0)

    # Define the list of FINAL generated features before dropping NaNs
    final_feature_set = [col for col in df_feat.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']]
    # Check that all generated features actually exist before using in subset
    final_feature_set = [col for col in final_feature_set if col in df_feat.columns] 

    # --- Handle NaNs ---
    print("Handling NaNs created by feature engineering...")
    initial_len = len(df_feat)
    df_feat.dropna(subset=final_feature_set, inplace=True) 
    final_len = len(df_feat)
    print(f"Rows after dropping initial NaNs based on features: {final_len} (dropped {initial_len-final_len})")
    
    if final_len == 0: return pd.DataFrame()

    print(f"Finished generating {len(final_feature_set)} features. Final shape before target: {df_feat.shape}")
    return df_feat


# --- Target Definition Function ---
def add_target(df, lookahead=LOOKAHEAD_PERIOD, gain=TARGET_GAIN):
    print(f"Adding target variable (lookahead={lookahead} mins, gain=${gain})...")
    if df.empty: print("Cannot add target to empty DataFrame."); return df
    df_target = df.copy()
    if 'close' not in df_target.columns: print("Error: 'close' column missing."); return pd.DataFrame() 

    df_target['future_price'] = df_target['close'].shift(-lookahead)
    df_target['target'] = ((df_target['future_price'] - df_target['close']) >= gain).astype(int)
    
    initial_len = len(df_target)
    df_target.dropna(subset=['future_price', 'target'], inplace=True) 
    final_len = len(df_target)
    print(f"Rows after dropping rows without future target: {final_len} (removed {initial_len - final_len})")
    return df_target


# --- Backtesting Function Definition (Specific for XGBoost) ---
def backtest_xgb_model(df_with_features, xgb_model, fitted_scaler, feature_cols, lookahead=LOOKAHEAD_PERIOD, gain_threshold=TARGET_GAIN, prob_threshold=PREDICTION_THRESHOLD):
    print("\nScaling full feature set for backtest...")
    missing_cols = [col for col in feature_cols if col not in df_with_features.columns]
    if missing_cols: print(f"Error: Missing backtest features: {missing_cols}"); return pd.DataFrame()

    # Select only the necessary features before scaling
    X_features = df_with_features[feature_cols].values
    X_full_scaled = fitted_scaler.transform(X_features)

    results = []
    iteration_end = len(df_with_features) - lookahead
    print(f"Iterating for backtest (predicting for index 0 to {iteration_end - 1})...") 
    num_predictions = 0

    for i in range(iteration_end): 
        current_features_scaled = X_full_scaled[i:i+1] 
        prob = 0.0
        try:
            prob = xgb_model.predict_proba(current_features_scaled)[0, 1] 
            num_predictions += 1
        except Exception as e: print(f"Error predicting at step index {i}: {e}"); continue

        if i % 500 == 0: print(f"  Backtest progress: Predicted step index {i}...")

        decision_timestamp = df_with_features.index[i]
        outcome_timestamp = df_with_features.index[i + lookahead]
        current_price_at_decision = df_with_features.loc[decision_timestamp, 'close']
        future_price_at_outcome = df_with_features.loc[outcome_timestamp, 'close'] 
        actual_gain = future_price_at_outcome - current_price_at_decision
        actual_outcome_met_target = actual_gain >= gain_threshold
        would_trade = prob >= prob_threshold
        prediction_correct = (would_trade == actual_outcome_met_target)

        results.append({
            'timestamp': decision_timestamp, 'current_price': current_price_at_decision, 
            'future_price': future_price_at_outcome, 'probability': prob, 'actual_gain': actual_gain,
            'would_trade': would_trade, 'actual_outcome': actual_outcome_met_target, 
            'prediction_correct': prediction_correct   
        })

    print(f"Finished backtest loop, {num_predictions} predictions made.")
    return pd.DataFrame(results)

# ==============================================================================
#                             MAIN EXECUTION FLOW
# ==============================================================================

# 1. Create Features
features_only_df = create_features(df)
if features_only_df.empty: print("Feature generation failed."); exit(1)

# 2. Add Target Variable
feature_df = add_target(features_only_df) 
if feature_df.empty: print("Target generation failed."); exit(1)

print(f"\nFinal DataFrame shape for modeling: {feature_df.shape}")
print(f"Target value counts:\n{feature_df['target'].value_counts(normalize=True)}")

# --- Data Preparation ---
print("\n--- Data Preparation for XGBoost ---")
# Re-select features based on final available columns after feature engineering
available_features = [col for col in features_only_df.columns if col not in EXCLUDE_FROM_FEATURES]
features_to_use = [col for col in available_features if col in feature_df.columns] 
print(f"Selected {len(features_to_use)} features for model input: {features_to_use}")

X = feature_df[features_to_use] 
y = feature_df['target']        

# --- Train/Test Split (Time Series Style - Manual Slice) ---
print(f"Splitting data (Test size: {TEST_SET_SIZE})...")
split_index = int(len(X) * (1 - TEST_SET_SIZE))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:] 
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
if X_train.empty or X_test.empty: print("Train/Test split resulted in empty data."); exit(1)

# --- Scaling (Fit ONLY on Train) ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values) 
X_test_scaled = scaler.transform(X_test.values)     
print("Scaling complete.")

# --- XGBoost Model Training ---
print("\n--- Building and Training XGBoost Model ---")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

# Initialize model
model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss', 
    # ** For GPU Training (ensure xgboost compiled with GPU support) **
    # tree_method='hist', # Often needed for GPU
    # device='cuda',    # Specify GPU device
    n_estimators=200, learning_rate=0.05, max_depth=5, 
    subsample=0.8, colsample_bytree=0.8, gamma=0.1,               
    scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
    # --- Early Stopping arguments need to be passed HERE ---
    early_stopping_rounds=EARLY_STOPPING_ROUNDS_XGB # Use constant
)

print("Training XGBoost model with early stopping...")
# Evaluation set for early stopping
eval_set = [(X_test_scaled, y_test)] 

# Train the model, passing eval_set correctly
# The early_stopping_rounds argument is part of the *constructor* for XGBClassifier,
# not passed directly to fit() when using the scikit-learn API wrapper like this.
# The 'verbose' controls output DURING fit, but early stopping is configured at init.
model.fit(X_train_scaled, y_train,
          eval_set=eval_set,              # Monitor performance on this set 
          verbose=False)                  # Keep training output clean 
          
print("Training complete.")
# Note: Accessing best_iteration/score requires verbose=True during fit or 
# checking the evaluation history manually if eval_metric was specified during init.
# For now, we just know it trained and potentially stopped early based on eval_set performance.
print(f"Check verbose logs or model attributes for specific stopping iteration if needed.")


# --- Evaluate Model ---
print("\n--- Evaluating XGBoost Model ---")
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] 
y_pred_binary = (y_pred_proba > PREDICTION_THRESHOLD).astype(int) 

print(f"\nXGBoost Model Performance (Test Set, Threshold={PREDICTION_THRESHOLD}):")
print(classification_report(y_test, y_pred_binary, zero_division=0))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_binary)
try:
    plt.figure(figsize=(8, 6)); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (XGBoost)'); plt.colorbar(); unique_targets = np.unique(y_test); 
    tick_marks = np.arange(len(unique_targets)); plt.xticks(tick_marks, unique_targets); plt.yticks(tick_marks, unique_targets)
    plt.xlabel('Predicted label'); plt.ylabel('True label')
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape): plt.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout(); plt.savefig(CONFUSION_MATRIX_FILENAME); plt.close()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_FILENAME}")
except Exception as e: print(f"Error saving confusion matrix plot: {e}")


# --- Feature Importance ---
print("\n--- XGBoost Feature Importance ---")
try:
    # Ensure feature names list matches the columns in X_train
    feature_names_list = X_train.columns.tolist() 
    if len(feature_names_list) == len(model.feature_importances_):
        feature_importance = pd.DataFrame({'feature': feature_names_list, 'importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("(Top 15):"); print(feature_importance.head(15))
        
        plt.figure(figsize=(10, 8)); plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
        plt.xlabel("XGBoost Feature Importance"); plt.ylabel("Feature"); plt.title("Top 15 Feature Importances")
        plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(FEATURE_IMPORTANCE_FILENAME); plt.close()
        print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_FILENAME}")
    else:
         print(f"Error: Mismatch between number of feature names ({len(feature_names_list)}) and importances ({len(model.feature_importances_)})")

except Exception as e: print(f"Error plotting/processing feature importance: {e}")


# --- Save Artifacts ---
print("\n--- Saving Artifacts (XGBoost Model) ---")
try:
    # Save XGBoost model, scaler, and the ACTUAL list of feature names used
    joblib.dump((model, scaler, features_to_use), MODEL_SAVE_PATH)
    print(f"XGBoost model, scaler, and feature list saved to {MODEL_SAVE_PATH}")
except Exception as e: print(f"Error saving artifacts: {e}")


# --- Run XGBoost Backtest ---
print("\n--- Running Basic XGBoost Backtest ---")
if not feature_df.empty:
    xgb_backtest_results = backtest_xgb_model( 
        df_with_features=feature_df, 
        xgb_model=model,           
        fitted_scaler=scaler,        
        feature_cols=features_to_use 
    )

    # --- Analyze XGBoost Backtest Results ---
    print("\n--- XGBoost Backtest Analysis ---")
    if not xgb_backtest_results.empty:
        print(f"Total backtest prediction points: {len(xgb_backtest_results)}")
        xgb_trades = xgb_backtest_results[xgb_backtest_results['would_trade']].copy()
        print(f"'Trade' signals generated (prob >= {PREDICTION_THRESHOLD}): {len(xgb_trades)} ({ (len(xgb_trades)/len(xgb_backtest_results)*100) if len(xgb_backtest_results)>0 else 0 :.2f}%)") 
        if not xgb_trades.empty:
            winning_xgb_trades = xgb_trades[xgb_trades['actual_outcome'] == True] 
            print(f"Successful 'Trades' (Signal matched ${TARGET_GAIN:.2f}+ gain): {len(winning_xgb_trades)} ({(len(winning_xgb_trades)/len(xgb_trades)*100) if len(xgb_trades)>0 else 0:.2f}%)")
            print(f"Average actual gain/loss on signaled 'trades': ${xgb_trades['actual_gain'].mean():.2f}")
            print(f"Average actual gain on WINNING signaled 'trades': ${winning_xgb_trades['actual_gain'].mean() if not winning_xgb_trades.empty else 0:.2f}")

            # --- Visualize XGBoost Backtest ---
            print("\n--- Visualizing XGBoost Backtest ---")
            try:
                plt.figure(figsize=(15, 8))
                plt.plot(feature_df.index, feature_df['close'], 'k-', alpha=0.3, label=f'{TICKER} Price')
                if not winning_xgb_trades.empty: plt.scatter(winning_xgb_trades['timestamp'], winning_xgb_trades['current_price'], color='lime', marker='^', s=50, label='Successful Signal')
                losing_xgb_trades = xgb_trades[xgb_trades['actual_outcome'] == False]
                if not losing_xgb_trades.empty: plt.scatter(losing_xgb_trades['timestamp'], losing_xgb_trades['current_price'], color='red', marker='v', s=50, label='Unsuccessful Signal')
                plt.title(f'XGBoost Backtest Results ({TICKER}): Buy Signals (Prob>={PREDICTION_THRESHOLD}) & Outcomes'); plt.legend(); plt.ylabel("Price ($)"); plt.xlabel("Time"); plt.savefig(BACKTEST_PLOT_FILENAME); plt.close()
                print(f"Saved XGBoost backtest visualization to {BACKTEST_PLOT_FILENAME}")
            except Exception as e: print(f"Error plotting XGBoost backtest results: {e}")
        else: print("No 'trade' signals generated during backtest.")
    else: print("XGBoost Backtest produced no results.")
else: print("Skipping backtest as feature_df was empty.")

print("\nScript Finished.")
