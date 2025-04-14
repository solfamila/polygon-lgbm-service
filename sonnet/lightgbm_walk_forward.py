# --- Necessary Imports ---
import psycopg2
import pandas as pd
import numpy as np
import time
import os
import math 
from datetime import datetime, timedelta, timezone 
from dotenv import load_dotenv
import gc 

# --- NEW: LightGBM Import ---
import lightgbm as lgb

# ML/DL & Utility Imports 
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
DB_HOST = "localhost"; DB_PORT = "5433"; DB_NAME = "polygondata"; DB_USER = "polygonuser"
TICKER = "TSLA"

# Parameters 
LOOKAHEAD_PERIOD = 15; TARGET_GAIN = 1.0; TEST_SET_SIZE = 0.2     
THRESHOLDS_TO_TEST = [0.50, 0.55, 0.60, 0.65, 0.70] # Evaluate multiple thresholds
PREDICTION_THRESHOLD = 0.60 #<--- MAKE SURE THIS LINE EXISTS AND IS SPELLED CORRECTLY
# Walk-Forward Parameters
INITIAL_TRAIN_PCT = 0.60; ROLLING_WINDOW_PCT = 0.10 
# EARLY_STOPPING_ROUNDS_LGBM = 20 # LightGBM uses callbacks for early stopping

# Features to exclude
EXCLUDE_FROM_FEATURES = ['target', 'future_price', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']

# Output filenames (Adjusted for LightGBM)
MODEL_SAVE_PATH = f'lgbm_model_enhanced_{TICKER.lower()}.joblib' 
SCALER_COLS_FILENAME = f'lgbm_scaler_columns_enhanced_{TICKER.lower()}.joblib' # For consistency
CONFUSION_MATRIX_FILENAME = f'lgbm_wf_confusion_matrix_{TICKER.lower()}.png'
FEATURE_IMPORTANCE_FILENAME = f'lgbm_wf_feat_importance_{TICKER.lower()}.png' 
BACKTEST_PLOT_BASEFILENAME = f'lgbm_wf_backtest_{TICKER.lower()}' 


# --- Database Connection & Data Loading ---
# (Keep this section exactly as before)
# ... (Exact DB loading code here) ...
conn = None; df = pd.DataFrame() 
try:
    print(f"\nConnecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    print("Connection successful.")
    print(f"Fetching historical data for {TICKER}...")
    query = """SELECT start_time AS time, agg_open AS open, agg_high AS high, agg_low AS low, agg_close AS close, volume, vwap, num_trades FROM stock_aggregates_min WHERE symbol = %(ticker)s AND start_time IS NOT NULL ORDER BY start_time ASC;"""
    df = pd.read_sql(query, conn, params={'ticker': TICKER}, index_col='time', parse_dates={'time': {'utc': True}})
    print(f"Loaded {len(df)} historical bars for {TICKER} from DB.")
except Exception as e: print(f"Data loading error: {e}"); exit(1)
finally:
    if conn: conn.close(); print("Database connection closed.")
if df.empty: print("No data loaded."); exit(1)

# --- Enhanced Feature Engineering Function ---
# (Keep this section exactly as before - use the ENHANCED create_features)
# ... create_features function definition ...
def create_features(df): 
    # ... (exact enhanced function content) ...
    print("Generating features...")
    required_cols = ['open', 'high', 'low', 'close', 'volume']; required_ta_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols): print(f"Error: Missing required columns: {required_cols}"); return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
         try: df.index = pd.to_datetime(df.index, utc=True); print("Converted index.")
         except Exception as e: print(f"Error converting index: {e}"); return pd.DataFrame()
    if not df.index.is_monotonic_increasing: print("Warning: Sorting index..."); df.sort_index(inplace=True)
    df_feat = df.copy(); initial_rows = len(df_feat)
    # 1. Basic Features
    df_feat['return'] = df_feat['close'].pct_change()
    df_feat['high_low_range'] = df_feat['high'] - df_feat['low']
    df_feat['close_open_diff'] = df_feat['close'] - df_feat['open']
    df_feat['log_volume'] = np.log1p(df_feat['volume'].replace(0, 1)) 
    df_feat['price_vwap_diff'] = df_feat['close'] - df_feat['vwap'] if 'vwap' in df_feat.columns else 0
    df_feat['price_vwap_diff'] = df_feat['price_vwap_diff'].fillna(0) 
    # 2. MAs & Ratios
    for window in [5, 10, 20, 60]:
        df_feat[f'ma_{window}'] = df_feat['close'].rolling(window=window, min_periods=window).mean()
        df_feat[f'ma_vol_{window}'] = df_feat['volume'].rolling(window=window, min_periods=window).mean()
        df_feat[f'close_ma_{window}_ratio'] = (df_feat['close'] / df_feat[f'ma_{window}']).replace([np.inf, -np.inf], np.nan)
        df_feat[f'volume_ma_{window}_ratio'] = (df_feat['volume'] / df_feat[f'ma_vol_{window}']).replace([np.inf, -np.inf], np.nan)
    # 3. ROC
    for window in [1, 5, 10, 20]: df_feat[f'roc_{window}'] = df_feat['close'].pct_change(periods=window)
    # 4. Volatility
    df_feat['volatility_20'] = df_feat['return'].rolling(window=20, min_periods=20).std()
    # TA Features
    ta_cols_to_add = ['ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'RSI_14']
    for col in ta_cols_to_add: df_feat[col] = np.nan # Initialize
    if PANDAS_TA_AVAILABLE:
        print("Calculating pandas_ta features (ATR, Bollinger, RSI)...")
        try:
            if not all(col in df_feat.columns for col in required_ta_cols): raise ValueError(f"Missing columns for pandas_ta: {required_ta_cols}")
            df_feat.ta.atr(length=14, append=True); bbands_df = df_feat.ta.bbands(length=20, std=2, append=False); df_feat.ta.rsi(length=14, append=True)
            if bbands_df is not None and not bbands_df.empty: df_feat.update(bbands_df)
            print("TA features added.")
        except Exception as e: print(f"Warning: Error calculating pandas_ta features: {e}")
    # 5. Volume delta
    df_feat['volume_delta'] = df_feat['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
    # 6. Time features
    df_feat['hour'] = df_feat.index.hour; df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour']/24.0); df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour']/24.0)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['dayofweek']/7.0); df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['dayofweek']/7.0)
    # Final NaN Drop
    final_feature_set = [col for col in df_feat.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']]
    final_feature_set = [col for col in final_feature_set if col in df_feat.columns]
    df_feat.dropna(subset=final_feature_set, inplace=True)
    print(f"Finished features. Rows dropped: {initial_rows - len(df_feat)} / {initial_rows}. Final shape before target: {df_feat.shape}")
    return df_feat


# --- Target Definition Function ---
# (Keep this section exactly as before)
# ... add_target function definition ...
def add_target(df, lookahead=LOOKAHEAD_PERIOD, gain=TARGET_GAIN):
    print(f"Adding target variable (lookahead={lookahead} mins, gain=${gain})...")
    if df.empty: 
        print("Input DataFrame is empty, cannot add target.")
        return df # Return the empty DataFrame
    
    # --- FIX: Check 'close' column on the input 'df' first ---
    if 'close' not in df.columns: 
        print("Error: 'close' column missing from input DataFrame.")
        return pd.DataFrame() # Return empty if required column missing
    
    # --- Now it's safe to copy ---
    df_target = df.copy() 

    df_target['future_price'] = df_target['close'].shift(-lookahead)
    df_target['target'] = ((df_target['future_price'] - df_target['close']) >= gain).astype(int)
    
    initial_len = len(df_target)
    # Drop rows where the target could not be calculated 
    df_target.dropna(subset=['future_price', 'target'], inplace=True) 
    final_len = len(df_target)
    print(f"Rows after dropping future target NaNs: {final_len} (removed {initial_len - final_len})")
    
    # Add check for empty df after dropna
    if final_len == 0:
        print("Warning: All rows dropped after adding target (check lookahead period vs data length).")
        
    return df_target

# --- Backtesting Function Definition (Adapted for LightGBM) ---
# We need to modify this slightly as LightGBM API is a bit different for predict_proba
def backtest_lgbm_model(df_with_features, lgbm_model, fitted_scaler, feature_cols, lookahead=LOOKAHEAD_PERIOD, gain_threshold=TARGET_GAIN, prob_threshold=PREDICTION_THRESHOLD):
    print("\nScaling full feature set for backtest...")
    missing_cols = [col for col in feature_cols if col not in df_with_features.columns]
    if missing_cols: print(f"Error: Missing backtest features: {missing_cols}"); return pd.DataFrame()
    X_features = df_with_features[feature_cols].values
    X_full_scaled = fitted_scaler.transform(X_features) # Scale necessary features
    
    results = []
    iteration_end = len(df_with_features) - lookahead
    print(f"Iterating for backtest (predicting for index 0 to {iteration_end - 1})...") 
    num_predictions = 0

    for i in range(iteration_end): 
        current_features_scaled = X_full_scaled[i:i+1] # Input as 2D row
        prob = 0.0
        try:
            # LightGBM predict_proba returns shape (n_samples, n_classes)
            prob = lgbm_model.predict_proba(current_features_scaled)[0, 1] # Probability of class 1
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
#                   WALK-FORWARD VALIDATION - LightGBM
# ==============================================================================
print("\n--- Starting Walk-Forward Validation (LightGBM) ---")

# 1. Prepare Data (Reuse from before)
print("Step 1: Preparing full dataset with features and target...")
features_only_df = create_features(df)
if features_only_df.empty: exit("Feature generation failed.")
feature_df = add_target(features_only_df) 
if feature_df.empty: exit("Target generation failed."); exit(1)

features_to_use = [col for col in feature_df.columns if col not in EXCLUDE_FROM_FEATURES]
print(f"Using {len(features_to_use)} features for modeling: {features_to_use}")

X_full = feature_df[features_to_use]
y_full = feature_df['target']

min_data_needed = 100 
if len(feature_df) < min_data_needed: exit(f"Error: Not enough data ({len(feature_df)}) for walk-forward.")

# --- Walk-Forward Splitting & Loop ---
n_samples = len(X_full)
initial_train_size = int(n_samples * INITIAL_TRAIN_PCT)
rolling_window_size = int(n_samples * ROLLING_WINDOW_PCT) 
if rolling_window_size < 1: rolling_window_size = 1 
print(f"Total samples: {n_samples}, Initial train: {initial_train_size}, Predict window: {rolling_window_size}")

all_test_indices_original = []; all_predictions_proba = []; all_actuals = []
all_feature_importances = [] # Store LightGBM importance

scaler = StandardScaler() 
fold = 0
current_train_end_index = initial_train_size

while current_train_end_index < n_samples:
    fold += 1
    train_indices = np.arange(0, current_train_end_index)
    test_indices = np.arange(current_train_end_index, min(current_train_end_index + rolling_window_size, n_samples))
    if len(test_indices) == 0: break 

    print(f"\n--- Fold {fold} ---")
    print(f"  Training indices: {train_indices[0]}-{train_indices[-1]} ({len(train_indices)}), Test indices: {test_indices[0]}-{test_indices[-1]} ({len(test_indices)})")
    
    X_train_fold, y_train_fold = X_full.iloc[train_indices].values, y_full.iloc[train_indices].values
    X_test_fold, y_test_fold = X_full.iloc[test_indices].values, y_full.iloc[test_indices].values
    original_test_fold_indices = X_full.iloc[test_indices].index 

    X_train_fold_scaled = scaler.fit_transform(X_train_fold) 
    X_test_fold_scaled = scaler.transform(X_test_fold)

    # LightGBM uses 'is_unbalance' or 'scale_pos_weight'
    lgbm_scale_pos_weight = (y_train_fold == 0).sum() / (y_train_fold == 1).sum() if (y_train_fold == 1).sum() > 0 else 1
    print(f"  Scale pos weight for fold: {lgbm_scale_pos_weight:.2f}")

    # --- Initialize and Train LightGBM Model ---
    # ** Define eval_set for early stopping using the TEST data of the FOLD **
    eval_set_fold = [(X_test_fold_scaled, y_test_fold)]
    
    model_fold = lgb.LGBMClassifier(
        objective='binary',        # Binary classification objective
        metric='logloss',           # Metric to monitor for early stopping
        n_estimators=1000,         # Potentially more estimators, rely on early stopping
        learning_rate=0.05,
        num_leaves=31,             # Standard parameter for LGBM depth/complexity
        # is_unbalance=True,       # Alternative to scale_pos_weight
        scale_pos_weight=lgbm_scale_pos_weight, # Use calculated weight
        random_state=42,
        n_jobs=-1
        # reg_alpha=0.1, reg_lambda=0.1 # Optional regularization
    )
    
    print(f"  Training LightGBM model for fold {fold}...")
    
    # ** Early stopping with LightGBM requires using callbacks **
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)] # Stop if logloss doesn't improve
    
    # Train model
    model_fold.fit(
        X_train_fold_scaled, y_train_fold,
        eval_set=eval_set_fold,       
        eval_metric='logloss', # <-- ADD THIS LINE
        callbacks=callbacks          
        # Note: verbose controls console output, not directly related to callback operation
    )
    
    if hasattr(model_fold, 'feature_importances_'): 
        all_feature_importances.append(model_fold.feature_importances_)

    print(f"  Predicting on test fold {fold}...")
    y_pred_proba_fold = model_fold.predict_proba(X_test_fold_scaled)[:, 1]
    
    all_test_indices_original.extend(original_test_fold_indices) 
    all_predictions_proba.extend(y_pred_proba_fold)
    all_actuals.extend(y_test_fold) 

    current_train_end_index = test_indices[-1] + 1 
    del X_train_fold, y_train_fold, X_test_fold, y_test_fold, X_train_fold_scaled, X_test_fold_scaled, model_fold; gc.collect() 


# --- Combine and Evaluate Walk-Forward Results ---
print("\n--- Aggregated Walk-Forward Validation Results (LightGBM) ---")
if not all_actuals:
     print("No predictions generated.")
else:
    print(f"Total out-of-sample predictions made: {len(all_actuals)}")
    wf_results_df = pd.DataFrame({
         'timestamp': all_test_indices_original, 'probability': all_predictions_proba, 'actual': all_actuals
    }).set_index('timestamp') 

    print("\nOverall Performance across Thresholds:")
    print("-" * 60)
    for threshold in THRESHOLDS_TO_TEST:
         print(f"Threshold = {threshold:.2f}")
         wf_results_df[f'prediction'] = (wf_results_df['probability'] >= threshold).astype(int) # Use dynamic name
         print(classification_report(wf_results_df['actual'], wf_results_df[f'prediction'], zero_division=0))
         print("-" * 60)

    # Confusion Matrix for the baseline threshold
    baseline_predictions = (wf_results_df['probability'] >= PREDICTION_THRESHOLD).astype(int)
    cm_wf = confusion_matrix(wf_results_df['actual'], baseline_predictions)
    try: 
        plt.figure(figsize=(8, 6)); plt.imshow(cm_wf, interpolation='nearest', cmap=plt.cm.Blues); plt.title(f'Walk-Forward CM (LGBM, Thresh={PREDICTION_THRESHOLD:.2f})')
        plt.colorbar(); unique_targets = np.unique(all_actuals); tick_marks = np.arange(len(unique_targets)); 
        plt.xticks(tick_marks, unique_targets); plt.yticks(tick_marks, unique_targets)
        plt.xlabel('Predicted label'); plt.ylabel('True label'); thresh = cm_wf.max() / 2.
        for i, j in np.ndindex(cm_wf.shape): plt.text(j, i, f'{cm_wf[i, j]}', ha='center', va='center', color='white' if cm_wf[i, j] > thresh else 'black')
        plt.tight_layout(); plt.savefig(CONFUSION_MATRIX_FILENAME); plt.close()
        print(f"Walk-forward confusion matrix (thresh={PREDICTION_THRESHOLD:.2f}) saved to {CONFUSION_MATRIX_FILENAME}")
    except Exception as e: print(f"Error saving confusion matrix plot: {e}")

    # Aggregate Feature Importances
    if all_feature_importances:
         print("\n--- Aggregated Feature Importance (Mean over Folds - LightGBM) ---")
         try:
             avg_importance = np.mean(all_feature_importances, axis=0)
             if len(features_to_use) == len(avg_importance):
                 feature_importance_agg = pd.DataFrame({'feature': features_to_use, 'importance': avg_importance}).sort_values('importance', ascending=False)
                 print("(Top 15):"); print(feature_importance_agg.head(15))
                 plt.figure(figsize=(10, 8)); plt.barh(feature_importance_agg['feature'][:15], feature_importance_agg['importance'][:15])
                 plt.xlabel("Mean LGBM Feature Importance (Walk-Forward)"); plt.ylabel("Feature"); plt.title("Avg Top 15 Feature Importances")
                 plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(FEATURE_IMPORTANCE_FILENAME); plt.close()
                 print(f"Aggregated feature importance plot saved to {FEATURE_IMPORTANCE_FILENAME}")
             else: print("Error: Feature name count != Importance count")
         except Exception as e: print(f"Error processing feature importance: {e}")

    # --- Generate Backtest Analysis from Walk-Forward Predictions ---
    print("\n--- Generating Backtest Analysis from Walk-Forward Predictions (LightGBM) ---")
    wf_backtest_df = wf_results_df.join(feature_df[['close', 'future_price']], how='inner')
    
    if 'close' not in wf_backtest_df.columns or 'future_price' not in wf_backtest_df.columns:
        print("Error: Could not merge price data for backtest analysis.")
    else:
        wf_backtest_df.rename(columns={'close': 'current_price'}, inplace=True)
        wf_backtest_df['actual_gain'] = wf_backtest_df['future_price'] - wf_backtest_df['current_price']
        wf_backtest_df['actual_outcome'] = (wf_backtest_df['actual_gain'] >= TARGET_GAIN)
        
        # --- Analyze Backtest for Different Thresholds ---
        print("\n--- Walk-Forward Backtest Analysis (Multiple Thresholds - LightGBM) ---")
        print("-" * 70)
        for threshold in THRESHOLDS_TO_TEST:
             wf_backtest_df['would_trade'] = (wf_backtest_df['probability'] >= threshold) # Recalculate based on threshold
             
             print(f"Results for Threshold = {threshold:.2f}")
             
             trades = wf_backtest_df[wf_backtest_df['would_trade']].copy()
             print(f"  'Trade' signals generated: {len(trades)} ({ (len(trades)/len(wf_backtest_df)*100) if len(wf_backtest_df)>0 else 0 :.2f}%)") 
             if not trades.empty:
                 winning_trades = trades[trades['actual_outcome'] == True] 
                 print(f"  Successful 'Trades' (Matched Target): {len(winning_trades)} ({(len(winning_trades)/len(trades)*100) if len(trades)>0 else 0:.2f}%)")
                 print(f"  Avg Gain/Loss on Signals: ${trades['actual_gain'].mean():.2f}")
                 print(f"  Avg Gain on Successful Signals: ${winning_trades['actual_gain'].mean() if not winning_trades.empty else 0:.2f}")

                 if threshold == PREDICTION_THRESHOLD: 
                      # --- Visualize Baseline Threshold Walk-Forward Backtest ---
                      print(f"  Visualizing Backtest for threshold {threshold:.2f}...")
                      try:
                           plt.figure(figsize=(15, 8))
                           plt.plot(feature_df.index, feature_df['close'], 'k-', alpha=0.3, label=f'{TICKER} Price') 
                           if not winning_trades.empty: plt.scatter(winning_trades.index, winning_trades['current_price'], color='lime', marker='^', s=30, label='Successful Signal')
                           losing_trades = trades[trades['actual_outcome'] == False]
                           if not losing_trades.empty: plt.scatter(losing_trades.index, losing_trades['current_price'], color='red', marker='v', s=30, label='Unsuccessful Signal')
                           plt.title(f'Walk-Forward LightGBM Backtest ({TICKER}): Signals (Prob>={threshold:.2f}) & Outcomes'); plt.legend(); plt.ylabel("Price ($)"); plt.xlabel("Time")
                           # Adjust filename for plot
                           plot_filename = BACKTEST_PLOT_BASEFILENAME + f'_thresh_{threshold:.2f}.png'
                           plt.savefig(plot_filename); plt.close()
                           print(f"  Saved WF backtest visualization to {plot_filename}")
                      except Exception as e: print(f"  Error plotting WF backtest: {e}")
             else: print("  No 'trade' signals generated for this threshold.")
             print("-" * 70)

# --- No final model training/saving - evaluation based on WF ---
print("\nWalk-Forward Validation Script Finished.")
