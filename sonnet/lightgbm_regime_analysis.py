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

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import joblib 

try:
    import pandas_ta as ta; PANDAS_TA_AVAILABLE = True; print("pandas_ta imported.")
except ImportError: PANDAS_TA_AVAILABLE = False; print("Warning: pandas_ta not installed.")

import warnings
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.*")

print("\n--- Configuration & Setup ---")

# --- Configuration Loading & Constants ---
load_dotenv(); DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST="localhost"; DB_PORT="5433"; DB_NAME="polygondata"; DB_USER="polygonuser"
TICKER = "TSLA"; LOOKAHEAD_PERIOD = 15; TARGET_GAIN = 1.0; TEST_SET_SIZE = 0.2     
THRESHOLDS_TO_TEST = [0.50, 0.55, 0.60, 0.65] # Adjusted thresholds
PREDICTION_THRESHOLD = 0.60
INITIAL_TRAIN_PCT = 0.60; ROLLING_WINDOW_PCT = 0.10 
# Features to exclude
EXCLUDE_FROM_FEATURES = ['target', 'future_price', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']

# <<< NEW: Regime Detection Parameters >>>
REGIME_VOL_WINDOW = 60 * 8 # Lookback window for volatility calculation (e.g., 8 hours)
REGIME_LOW_THRESH = 30    # VIX-like threshold for low vol (annualized %)
REGIME_MED_THRESH = 60    # VIX-like threshold for medium vol

# Output filenames (Adjusted for LightGBM)
MODEL_SAVE_PATH = f'lgbm_model_enhanced_{TICKER.lower()}.joblib' 
SCALER_COLS_FILENAME = f'lgbm_scaler_columns_enhanced_{TICKER.lower()}.joblib' 
CONFUSION_MATRIX_FILENAME = f'lgbm_wf_confusion_matrix_{TICKER.lower()}.png'
FEATURE_IMPORTANCE_FILENAME = f'lgbm_wf_feat_importance_{TICKER.lower()}.png' 
BACKTEST_PLOT_BASEFILENAME = f'lgbm_wf_backtest_{TICKER.lower()}' # <-- ADD THIS LINE (ensure correct name)

# Output filenames (Adjusted for regime analysis)
REGIME_CM_FILENAME_BASE = f'lgbm_wf_cm_{TICKER.lower()}' # Will add regime later
# Omitting aggregate importance/backtest plots for now, focusing on metrics
# FEATURE_IMPORTANCE_FILENAME = f'lgbm_wf_feat_importance_{TICKER.lower()}.png' 
# BACKTEST_PLOT_BASEFILENAME = f'lgbm_wf_backtest_{TICKER.lower()}' 


# --- Database Connection & Data Loading ---
# ... (Keep this section exactly as before) ...
conn = None; df = pd.DataFrame() 
try: # Load data... (same as before)
    print(f"\nConnecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    print("Connection successful."); print(f"Fetching historical data for {TICKER}...")
    query = """SELECT start_time AS time, agg_open AS open, agg_high AS high, agg_low AS low, agg_close AS close, volume, vwap, num_trades FROM stock_aggregates_min WHERE symbol = %(ticker)s AND start_time IS NOT NULL ORDER BY start_time ASC;"""
    df = pd.read_sql(query, conn, params={'ticker': TICKER}, index_col='time', parse_dates={'time': {'utc': True}})
    print(f"Loaded {len(df)} historical bars for {TICKER} from DB.")
except Exception as e: print(f"Data loading error: {e}"); exit(1)
finally:
    if conn: conn.close(); print("Database connection closed.")
if df.empty: print("No data loaded."); exit(1)


# --- Enhanced Feature Engineering Function ---
# ... (Keep create_features function exactly as before) ...
def create_features(df): 
    # ... (exact enhanced function content) ...
    print("Generating features..."); required_cols = ['open', 'high', 'low', 'close', 'volume']; required_ta_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols): print(f"Error: Missing cols: {required_cols}"); return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
         try: df.index = pd.to_datetime(df.index, utc=True); print("Converted index.")
         except Exception as e: print(f"Error converting index: {e}"); return pd.DataFrame()
    if not df.index.is_monotonic_increasing: print("Warning: Sorting index..."); df.sort_index(inplace=True)
    df_feat = df.copy(); initial_rows = len(df_feat)
    # 1. Basic
    df_feat['return'] = df_feat['close'].pct_change(); df_feat['high_low_range'] = df_feat['high'] - df_feat['low']
    df_feat['close_open_diff'] = df_feat['close'] - df_feat['open']; df_feat['log_volume'] = np.log1p(df_feat['volume'].replace(0, 1)) 
    df_feat['price_vwap_diff'] = df_feat['close'] - df_feat['vwap'] if 'vwap' in df_feat.columns else 0; df_feat['price_vwap_diff'] = df_feat['price_vwap_diff'].fillna(0) 
    # 2. MAs/Ratios
    for window in [5, 10, 20, 60]:
        df_feat[f'ma_{window}'] = df_feat['close'].rolling(window=window, min_periods=window).mean(); df_feat[f'ma_vol_{window}'] = df_feat['volume'].rolling(window=window, min_periods=window).mean()
        df_feat[f'close_ma_{window}_ratio'] = (df_feat['close'] / df_feat[f'ma_{window}']).replace([np.inf, -np.inf], np.nan); df_feat[f'volume_ma_{window}_ratio'] = (df_feat['volume'] / df_feat[f'ma_vol_{window}']).replace([np.inf, -np.inf], np.nan)
    # 3. ROC
    for window in [1, 5, 10, 20]: df_feat[f'roc_{window}'] = df_feat['close'].pct_change(periods=window)
    # 4. Volatility
    df_feat['volatility_20'] = df_feat['return'].rolling(window=20, min_periods=20).std()
    # TA Features
    ta_cols_to_add = ['ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'RSI_14']
    for col in ta_cols_to_add: df_feat[col] = np.nan
    if PANDAS_TA_AVAILABLE:
        print("Calculating pandas_ta features (ATR, Bollinger, RSI)...")
        try:
            if not all(col in df_feat.columns for col in required_ta_cols): raise ValueError(f"Missing columns for pandas_ta: {required_ta_cols}")
            df_feat.ta.atr(length=14, append=True); bbands_df = df_feat.ta.bbands(length=20, std=2, append=False); df_feat.ta.rsi(length=14, append=True)
            if bbands_df is not None and not bbands_df.empty: df_feat.update(bbands_df); print("TA features added.")
        except Exception as e: print(f"Warning: Error calculating pandas_ta features: {e}")
    # 5. Volume delta
    df_feat['volume_delta'] = df_feat['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
    # 6. Time features
    df_feat['hour'] = df_feat.index.hour; df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour']/24.0); df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour']/24.0)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['dayofweek']/7.0); df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['dayofweek']/7.0)
    # Final NaN Drop
    final_feature_set = [col for col in df_feat.columns if col not in EXCLUDE_FROM_FEATURES and col not in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']]
    final_feature_set = [col for col in final_feature_set if col in df_feat.columns]
    df_feat.dropna(subset=final_feature_set, inplace=True)
    print(f"Finished features. Rows dropped: {initial_rows - len(df_feat)} / {initial_rows}. Final shape before target: {df_feat.shape}")
    return df_feat


# --- Target Definition Function ---
# (Keep this section exactly as before)
def add_target(df, lookahead=LOOKAHEAD_PERIOD, gain=TARGET_GAIN):
    print(f"Adding target variable (lookahead={lookahead} mins, gain=${gain})...")
    if df.empty: 
        print("Input DataFrame is empty, cannot add target.")
        return df 
    
    # --- FIX: Check 'close' column on the input 'df' first ---
    if 'close' not in df.columns: 
        print("Error: 'close' column missing from input DataFrame.")
        return pd.DataFrame() 
    
    # --- Now it's safe to copy ---
    df_target = df.copy() 

    df_target['future_price'] = df_target['close'].shift(-lookahead)
    df_target['target'] = ((df_target['future_price'] - df_target['close']) >= gain).astype(int)
    
    initial_len = len(df_target)
    df_target.dropna(subset=['future_price', 'target'], inplace=True) 
    final_len = len(df_target)
    print(f"Rows after dropping future target NaNs: {final_len} (removed {initial_len - final_len})")
    
    if final_len == 0:
        print("Warning: All rows dropped after adding target.")
        
    return df_target

# --- <<< NEW: Regime Detection Function >>> ---
def detect_market_regime(price_series, window=REGIME_VOL_WINDOW, low_thresh=REGIME_LOW_THRESH, med_thresh=REGIME_MED_THRESH):
    """
    Detects market regimes based on annualized rolling volatility of minute returns.
    Assumes 252 trading days, 6.5 hours/day, 60 mins/hour for annualization factor.
    Adjust factor if your data includes non-market hours.
    """
    print(f"Detecting regimes using window={window}, low<{low_thresh} <med<{med_thresh} <high")
    # Calculate minute returns
    minute_returns = price_series.pct_change()
    # Calculate rolling standard deviation (volatility)
    rolling_std = minute_returns.rolling(window=window, min_periods=window // 2).std() # Need min_periods
    # Annualize (adjust factor based on your data frequency/coverage)
    # Market hours factor: sqrt(252 * 6.5 * 60) ~ sqrt(98280) ~ 313.5
    # 24/7 factor: sqrt(365 * 24 * 60) ~ sqrt(525600) ~ 725
    annualization_factor = 313.5 # Adjust if needed! Using market hours estimate
    annualized_vol = rolling_std * annualization_factor * 100 # In percent

    # Define regimes
    regimes = pd.Series(index=price_series.index, dtype=object) # Use object type initially
    regimes[annualized_vol <= low_thresh] = 'low_vol'
    regimes[(annualized_vol > low_thresh) & (annualized_vol <= med_thresh)] = 'medium_vol'
    regimes[annualized_vol > med_thresh] = 'high_vol'
    regimes.fillna('undefined', inplace=True) # Fill initial NaNs or gaps

    print("Regime detection complete.")
    return regimes

# ==============================================================================
#                   WALK-FORWARD VALIDATION IMPLEMENTATION
# ==============================================================================
print("\n--- Starting Walk-Forward Validation (LightGBM with Regime Analysis) ---")

# 1. Prepare Data (Apply features, target AND regime detection)
print("Step 1: Preparing full dataset...")
features_only_df = create_features(df)
if features_only_df.empty: exit("Feature generation failed.")
# --- <<< NEW: Add Regime Detection AFTER Feature Engineering >>> ---
features_only_df['regime'] = detect_market_regime(features_only_df['close'])
print("Regime value counts in feature set:")
print(features_only_df['regime'].value_counts())
# --- End New ---
feature_df = add_target(features_only_df) 
if feature_df.empty: exit("Target generation failed."); exit(1)

features_to_use = [col for col in feature_df.columns if col not in EXCLUDE_FROM_FEATURES + ['regime']] # Exclude regime from model features
print(f"Using {len(features_to_use)} features for modeling: {features_to_use}")

X_full = feature_df[features_to_use]
y_full = feature_df['target']
# <<< NEW: Keep regime column aligned with predictions >>>
regimes_full = feature_df['regime']

min_data_needed = 100 
if len(feature_df) < min_data_needed: exit(f"Error: Not enough data ({len(feature_df)}) for walk-forward.")

# --- Walk-Forward Splitting & Loop ---
# ... (Keep splitting logic and loop structure exactly as before) ...
n_samples = len(X_full); initial_train_size = int(n_samples * INITIAL_TRAIN_PCT)
rolling_window_size = int(n_samples * ROLLING_WINDOW_PCT); 
if rolling_window_size < 1: rolling_window_size = 1 
print(f"Total samples: {n_samples}, Initial train: {initial_train_size}, Predict window: {rolling_window_size}")

all_test_indices_original = []; all_predictions_proba = []; all_actuals = []; all_regimes = []
fold = 0; current_train_end_index = initial_train_size
scaler = StandardScaler() 

while current_train_end_index < n_samples:
    fold += 1; train_indices = np.arange(0, current_train_end_index)
    test_indices = np.arange(current_train_end_index, min(current_train_end_index + rolling_window_size, n_samples))
    if len(test_indices) == 0: break 

    print(f"\n--- Fold {fold} ---")
    # ... (Select fold data: X_train_fold, y_train_fold, X_test_fold, y_test_fold exactly as before) ...
    X_train_fold, y_train_fold = X_full.iloc[train_indices].values, y_full.iloc[train_indices].values
    X_test_fold, y_test_fold = X_full.iloc[test_indices].values, y_full.iloc[test_indices].values
    original_test_fold_indices = X_full.iloc[test_indices].index 
    # <<< NEW: Get regimes for the test fold >>>
    regimes_test_fold = regimes_full.iloc[test_indices].values 

    X_train_fold_scaled = scaler.fit_transform(X_train_fold) 
    X_test_fold_scaled = scaler.transform(X_test_fold)
    # Calculation line (already correct):
    scale_pos_weight = (y_train_fold == 0).sum() / (y_train_fold == 1).sum() if (y_train_fold == 1).sum() > 0 else 1
    print(f"  Scale pos weight for fold: {scale_pos_weight:.2f}")

    # --- Initialize and Train LightGBM Model ---
    model_fold = lgb.LGBMClassifier(
        objective='binary',        
        metric='logloss',           
        n_estimators=1000,         
        learning_rate=0.05,
        num_leaves=31,             
        # is_unbalance=True,       # Alternative to scale_pos_weight
        # scale_pos_weight=lgbm_scale_pos_weight, # <-- INCORRECT Variable Name
        scale_pos_weight=scale_pos_weight, # <-- CORRECT Variable Name
        random_state=42,
        n_jobs=-1
    )
    print(f"  Training LightGBM model fold {fold} (scale_pos_weight: {scale_pos_weight:.2f})...")
    eval_set_fold = [(X_test_fold_scaled, y_test_fold)]
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
    model_fold.fit(X_train_fold_scaled, y_train_fold, eval_set=eval_set_fold, eval_metric='logloss', callbacks=callbacks)
    
    # --- Predict and Store Results ---
    print(f"  Predicting on test fold {fold}...")
    y_pred_proba_fold = model_fold.predict_proba(X_test_fold_scaled)[:, 1]
    all_test_indices_original.extend(original_test_fold_indices) 
    all_predictions_proba.extend(y_pred_proba_fold)
    all_actuals.extend(y_test_fold) 
    # <<< NEW: Store regimes >>>
    all_regimes.extend(regimes_test_fold) 

    current_train_end_index = test_indices[-1] + 1 
    del X_train_fold, y_train_fold, X_test_fold, y_test_fold, X_train_fold_scaled, X_test_fold_scaled, model_fold; gc.collect() 

# --- <<< CHANGE: Combine and Evaluate Walk-Forward Results BY REGIME >>> ---
print("\n--- Aggregated Walk-Forward Validation Results by REGIME (LightGBM) ---")
if not all_actuals:
     print("No predictions generated.")
else:
    print(f"Total out-of-sample predictions made: {len(all_actuals)}")
    
    # Create results DataFrame including the regime
    wf_results_df = pd.DataFrame({
         'timestamp': all_test_indices_original,
         'probability': all_predictions_proba,
         'actual': all_actuals,
         'regime': all_regimes  # <<< Include regime
    }).set_index('timestamp') 

    print("\nPerformance across Thresholds BY REGIME:")
    print("=" * 70)

    # Loop through unique regimes found in the test predictions
    unique_regimes = wf_results_df['regime'].unique()
    if 'undefined' in unique_regimes: print("Note: 'undefined' regime includes initial period before volatility window is full.")
        
    for regime in sorted(unique_regimes):
        print(f"\n--- Regime: {regime} ---")
        regime_df = wf_results_df[wf_results_df['regime'] == regime]
        print(f"  Number of predictions in this regime: {len(regime_df)}")
        if len(regime_df) == 0: continue # Skip if no data for this regime
            
        print("-" * 60)
        for threshold in THRESHOLDS_TO_TEST:
            print(f"  Threshold = {threshold:.2f}")
            # Apply prediction threshold ONLY to the regime subset
            regime_df[f'prediction'] = (regime_df['probability'] >= threshold).astype(int) 
            
            # Calculate metrics for THIS regime subset
            try:
                # Inside the threshold loop in regime analysis section:
                prediction_col_name = f'prediction_{threshold:.2f}' # Use unique name for safety
                # Assign directly to avoid SettingWithCopyWarning
                regime_df = regime_df.assign(**{prediction_col_name: (regime_df['probability'] >= threshold).astype(int)}) 
                # Use this new column name in classification_report:
                report = classification_report(regime_df['actual'], regime_df[prediction_col_name], zero_division=0)
                print(report)
                 # Confusion Matrix (Optional: Save one per regime?)
                 # cm_wf = confusion_matrix(regime_df['actual'], regime_df[f'prediction'])
                 # ... plotting logic ... savefig(f'{REGIME_CM_FILENAME_BASE}_{regime}_thresh_{threshold:.2f}.png')
            except ValueError as ve:
                 print(f"    Could not generate classification report for threshold {threshold}: {ve}") # E.g., if only one class present
            print("-" * 60)
        print(f"--- End Regime: {regime} ---\n")
    print("=" * 70)
            

    # --- Backtest Analysis (Still use overall results, but could also be done per regime) ---
    print("\n--- Generating OVERALL Backtest Analysis from Walk-Forward Predictions ---")
    wf_backtest_df = wf_results_df.join(feature_df[['close', 'future_price']], how='inner') # Use overall results
    
    if 'close' not in wf_backtest_df.columns or 'future_price' not in wf_backtest_df.columns:
        print("Error: Could not merge price data for backtest analysis.")
    else:
        wf_backtest_df.rename(columns={'close': 'current_price'}, inplace=True)
        wf_backtest_df['actual_gain'] = wf_backtest_df['future_price'] - wf_backtest_df['current_price']
        wf_backtest_df['actual_outcome'] = (wf_backtest_df['actual_gain'] >= TARGET_GAIN)
        
        print("\n--- Walk-Forward Backtest Analysis (Multiple Thresholds - Overall) ---")
        print("-" * 70)
        # Store results per threshold for summary/comparison later if needed
        backtest_summary = {} 

        for threshold in THRESHOLDS_TO_TEST:
             wf_backtest_df['would_trade'] = (wf_backtest_df['probability'] >= threshold)
             
             print(f"Results for Threshold = {threshold:.2f}")
             
             trades = wf_backtest_df[wf_backtest_df['would_trade']].copy()
             trade_count = len(trades)
             signal_freq = (trade_count / len(wf_backtest_df) * 100) if len(wf_backtest_df) > 0 else 0
             print(f"  'Trade' signals generated: {trade_count} ({signal_freq:.2f}%)") 
             
             if not trades.empty:
                 winning_trades = trades[trades['actual_outcome'] == True] 
                 win_count = len(winning_trades)
                 win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
                 avg_gain_loss = trades['actual_gain'].mean()
                 avg_win_gain = winning_trades['actual_gain'].mean() if not winning_trades.empty else 0

                 print(f"  Successful 'Trades' (Matched Target): {win_count} ({win_rate:.2f}%)")
                 print(f"  Avg Gain/Loss on Signals: ${avg_gain_loss:.2f}")
                 print(f"  Avg Gain on Successful Signals: ${avg_win_gain:.2f}")

                 backtest_summary[threshold] = {'trades': trade_count, 'win_rate': win_rate, 'avg_pl': avg_gain_loss, 'avg_win': avg_win_gain}

                 # Plotting only for one representative threshold to avoid clutter
                 if threshold == PREDICTION_THRESHOLD: 
                      print(f"  Visualizing Backtest for baseline threshold {threshold:.2f}...")
                      try:
                           plt.figure(figsize=(15, 8))
                           plt.plot(feature_df.index, feature_df['close'], 'k-', alpha=0.3, label=f'{TICKER} Price') 
                           if not winning_trades.empty: plt.scatter(winning_trades.index, winning_trades['current_price'], color='lime', marker='^', s=30, label='Successful Signal')
                           losing_trades = trades[trades['actual_outcome'] == False]
                           if not losing_trades.empty: plt.scatter(losing_trades.index, losing_trades['current_price'], color='red', marker='v', s=30, label='Unsuccessful Signal')
                           plt.title(f'Walk-Forward LightGBM Backtest ({TICKER}): Signals (Prob>={threshold:.2f}) & Outcomes'); plt.legend(); plt.ylabel("Price ($)"); plt.xlabel("Time")
                           plot_filename = BACKTEST_PLOT_BASEFILENAME + f'_thresh_{threshold:.2f}.png'
                           plt.savefig(plot_filename); plt.close()
                           print(f"  Saved WF backtest visualization to {plot_filename}")
                      except Exception as e: print(f"  Error plotting WF backtest: {e}")
             else: 
                  print("  No 'trade' signals generated for this threshold.")
                  backtest_summary[threshold] = {'trades': 0, 'win_rate': 0, 'avg_pl': 0, 'avg_win': 0}
             print("-" * 70)
             
    # --- Optional: Print Summary Table of Backtest Metrics ---
    print("\nBacktest Metrics Summary Across Thresholds:")
    summary_df = pd.DataFrame.from_dict(backtest_summary, orient='index')
    print(summary_df)


# --- Final Model Training/Saving removed as focus is WF evaluation ---

print("\nWalk-Forward Regime Analysis Script Finished.")
