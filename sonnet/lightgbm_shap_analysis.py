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

# --- LightGBM Import ---
import lightgbm as lgb

# --- SHAP Import ---
import shap 

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
TICKER = "TSLA"; LOOKAHEAD_PERIOD = 15; TARGET_GAIN = 1.0   
TEST_SET_SIZE = 0.2    
THRESHOLDS_TO_TEST = [0.50, 0.55, 0.60, 0.65]; # Adjusted
PREDICTION_THRESHOLD = 0.60 # Baseline for plotting
INITIAL_TRAIN_PCT = 0.60; ROLLING_WINDOW_PCT = 0.10 
EARLY_STOPPING_ROUNDS_LGBM = 20 # Use this for LightGBM's callback

EXCLUDE_FROM_FEATURES = ['target', 'future_price', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']

# Output filenames (Adjusted for LightGBM & SHAP)
MODEL_SAVE_PATH = f'lgbm_final_model_enhanced_{TICKER.lower()}.joblib' # Final model save path
SCALER_COLS_FILENAME = f'lgbm_final_scaler_columns_enhanced_{TICKER.lower()}.joblib' # Final scaler/cols
CONFUSION_MATRIX_FILENAME = f'lgbm_wf_confusion_matrix_{TICKER.lower()}.png'
FEATURE_IMPORTANCE_FILENAME = f'lgbm_final_feat_importance_{TICKER.lower()}.png' # Importance from final model
SHAP_SUMMARY_PLOT_FILENAME = f'lgbm_shap_summary_{TICKER.lower()}.png'
BACKTEST_PLOT_BASEFILENAME = f'lgbm_wf_backtest_{TICKER.lower()}' 


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
    print("Generating features..."); required_cols=['open','high','low','close','volume']; required_ta_cols=['high','low','close','volume']
    if not all(col in df.columns for col in required_cols): return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
         try: df.index = pd.to_datetime(df.index, utc=True)
         except Exception as e: print(f"Error converting index: {e}"); return pd.DataFrame()
    if not df.index.is_monotonic_increasing: df.sort_index(inplace=True)
    df_feat = df.copy(); initial_rows = len(df_feat)
    df_feat['return'] = df_feat['close'].pct_change(); df_feat['high_low_range'] = df_feat['high'] - df_feat['low']; df_feat['close_open_diff'] = df_feat['close'] - df_feat['open']
    df_feat['log_volume'] = np.log1p(df_feat['volume'].replace(0, 1)); df_feat['price_vwap_diff'] = df_feat['close'] - df_feat['vwap'] if 'vwap' in df_feat.columns else 0; df_feat['price_vwap_diff'] = df_feat['price_vwap_diff'].fillna(0) 
    for window in [5, 10, 20, 60]:
        df_feat[f'ma_{window}'] = df_feat['close'].rolling(window=window, min_periods=window).mean(); df_feat[f'ma_vol_{window}'] = df_feat['volume'].rolling(window=window, min_periods=window).mean()
        df_feat[f'close_ma_{window}_ratio'] = (df_feat['close'] / df_feat[f'ma_{window}']).replace([np.inf, -np.inf], np.nan); df_feat[f'volume_ma_{window}_ratio'] = (df_feat['volume'] / df_feat[f'ma_vol_{window}']).replace([np.inf, -np.inf], np.nan)
    for window in [1, 5, 10, 20]: df_feat[f'roc_{window}'] = df_feat['close'].pct_change(periods=window)
    df_feat['volatility_20'] = df_feat['return'].rolling(window=20, min_periods=20).std()
    # TA Features (placeholders defined first)
    ta_cols_to_add = ['ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'RSI_14']
    
    # --- FIX: Initialize columns correctly ---
    for col in ta_cols_to_add:
         if col not in df_feat.columns: # Check if TA lib might have already added it
             df_feat[col] = np.nan 
    # --- End Fix ---

    if PANDAS_TA_AVAILABLE:
        try:
            if not all(col in df_feat.columns for col in required_ta_cols): raise ValueError("Missing cols for TA")
            df_feat.ta.atr(length=14, append=True); bbands_df = df_feat.ta.bbands(length=20, std=2, append=False); df_feat.ta.rsi(length=14, append=True)
            if bbands_df is not None and not bbands_df.empty: df_feat.update(bbands_df)
        except Exception as e: print(f"Warning: pandas_ta error: {e}")
    df_feat['volume_delta'] = df_feat['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
    df_feat['hour'] = df_feat.index.hour; df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour']/24.0); df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour']/24.0)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['dayofweek']/7.0); df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['dayofweek']/7.0)
    final_feature_set = [col for col in df_feat.columns if col not in EXCLUDE_FROM_FEATURES+['open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']]; final_feature_set = [col for col in final_feature_set if col in df_feat.columns]
    df_feat.dropna(subset=final_feature_set, inplace=True)
    print(f"Finished features. Rows dropped: {initial_rows - len(df_feat)} / {initial_rows}. Final shape before target: {df_feat.shape}"); return df_feat

# --- Target Definition Function ---
# ... (Keep add_target function exactly as before) ...
def add_target(df, lookahead=LOOKAHEAD_PERIOD, gain=TARGET_GAIN):
    print(f"Adding target variable (lookahead={lookahead} mins, gain=${gain})...")
    # --- FIX: Separate statements ---
    if df.empty: 
        print("Input DataFrame is empty, cannot add target.")
        return df # Return the empty DataFrame
    
    # Check 'close' column on the input 'df' first
    if 'close' not in df.columns: 
        print("Error: 'close' column missing from input DataFrame.")
        return pd.DataFrame() # Return empty if required column missing
    
    # Now copy the DataFrame
    df_target = df.copy() 
    # --- End of Fix ---

    df_target['future_price'] = df_target['close'].shift(-lookahead)
    df_target['target'] = ((df_target['future_price'] - df_target['close']) >= gain).astype(int)
    
    initial_len = len(df_target)
    df_target.dropna(subset=['future_price', 'target'], inplace=True) 
    final_len = len(df_target)
    print(f"Rows after dropping future target NaNs: {final_len} (removed {initial_len - final_len})")
    
    if final_len == 0:
        print("Warning: All rows dropped after adding target.")
        
    return df_target

# --- Backtesting Function Definition (Still uses LightGBM model) ---
# ... (Keep backtest_lgbm_model function exactly as before) ...
def backtest_lgbm_model(df_with_features, lgbm_model, fitted_scaler, feature_cols, lookahead=LOOKAHEAD_PERIOD, gain_threshold=TARGET_GAIN, prob_threshold=PREDICTION_THRESHOLD):
    # ... (exact function content) ...
    print("\nScaling features for backtest..."); missing_cols = [col for col in feature_cols if col not in df_with_features.columns]; 
    if missing_cols: print(f"Error: Missing backtest features: {missing_cols}"); return pd.DataFrame()
    X_features = df_with_features[feature_cols].values; X_full_scaled = fitted_scaler.transform(X_features); results = []
    iteration_end = len(df_with_features) - lookahead; print(f"Iterating for backtest (indices 0 to {iteration_end - 1})..."); num_predictions = 0
    for i in range(iteration_end): 
        current_features_scaled = X_full_scaled[i:i+1]; prob = 0.0
        try: prob = lgbm_model.predict_proba(current_features_scaled)[0, 1]; num_predictions += 1
        except Exception as e: print(f"Error predicting @ {i}: {e}"); continue
        if i % 500 == 0: print(f"  Backtest progress @ {i}...")
        decision_timestamp = df_with_features.index[i]; outcome_timestamp = df_with_features.index[i + lookahead]
        current_price = df_with_features.loc[decision_timestamp, 'close']; future_price = df_with_features.loc[outcome_timestamp, 'close'] 
        actual_gain = future_price - current_price; actual_outcome = actual_gain >= gain_threshold; would_trade = prob >= prob_threshold; prediction_correct = (would_trade == actual_outcome)
        results.append({'timestamp': decision_timestamp, 'current_price': current_price, 'future_price': future_price, 'probability': prob, 'actual_gain': actual_gain, 'would_trade': would_trade, 'actual_outcome': actual_outcome, 'prediction_correct': prediction_correct})
    print(f"Finished backtest loop: {num_predictions} predictions."); return pd.DataFrame(results)

# ==============================================================================
#                             MAIN EXECUTION FLOW
# ==============================================================================

# 1. Prepare Data
print("\n--- Starting Walk-Forward Validation (LightGBM + SHAP Analysis) ---")
features_only_df = create_features(df); 
if features_only_df.empty: exit("Feature gen failed.")
feature_df = add_target(features_only_df); 
if feature_df.empty: exit("Target gen failed.")
features_to_use = [col for col in feature_df.columns if col not in EXCLUDE_FROM_FEATURES]
X_full = feature_df[features_to_use]; y_full = feature_df['target']
if len(feature_df) < 100: exit("Not enough data.")

# --- Walk-Forward Loop ---
n_samples = len(X_full); initial_train_size = int(n_samples * INITIAL_TRAIN_PCT); rolling_window_size = int(n_samples * ROLLING_WINDOW_PCT) 
if rolling_window_size < 1: rolling_window_size = 1 
print(f"\nWF Params: Samples={n_samples}, InitialTrain={initial_train_size}, StepSize={rolling_window_size}")
all_test_indices_original = []; all_predictions_proba = []; all_actuals = []; all_feature_importances_lgb = [] # Store LGBM importances
fold = 0; current_train_end_index = initial_train_size; scaler_wf = StandardScaler() 
while current_train_end_index < n_samples:
    fold += 1; train_indices = np.arange(0, current_train_end_index); test_indices = np.arange(current_train_end_index, min(current_train_end_index + rolling_window_size, n_samples))
    if len(test_indices) == 0: break 
    print(f"\n--- Fold {fold}: Train {len(train_indices)}, Test {len(test_indices)} ---")
    X_train_fold, y_train_fold = X_full.iloc[train_indices].values, y_full.iloc[train_indices].values
    X_test_fold, y_test_fold = X_full.iloc[test_indices].values, y_full.iloc[test_indices].values
    original_test_fold_indices = X_full.iloc[test_indices].index 
    X_train_fold_scaled = scaler_wf.fit_transform(X_train_fold); X_test_fold_scaled = scaler_wf.transform(X_test_fold)
    lgbm_scale_pos_weight = (y_train_fold == 0).sum() / (y_train_fold == 1).sum() if (y_train_fold == 1).sum() > 0 else 1
    model_fold = lgb.LGBMClassifier( objective='binary', metric='logloss', n_estimators=1000, learning_rate=0.05, num_leaves=31, scale_pos_weight=lgbm_scale_pos_weight, random_state=42, n_jobs=-1)
    print(f"  Training LGBM fold {fold} (scale_pos_weight: {lgbm_scale_pos_weight:.2f})...")
    eval_set_fold = [(X_test_fold_scaled, y_test_fold)]; callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS_LGBM, verbose=False)] # Use constant
    model_fold.fit(X_train_fold_scaled, y_train_fold, eval_set=eval_set_fold, eval_metric='logloss', callbacks=callbacks)
    if hasattr(model_fold, 'feature_importances_'): all_feature_importances_lgb.append(model_fold.feature_importances_)
    print(f"  Predicting test fold {fold}..."); y_pred_proba_fold = model_fold.predict_proba(X_test_fold_scaled)[:, 1]
    all_test_indices_original.extend(original_test_fold_indices); all_predictions_proba.extend(y_pred_proba_fold); all_actuals.extend(y_test_fold) 
    current_train_end_index = test_indices[-1] + 1; del X_train_fold, y_train_fold, X_test_fold, y_test_fold, X_train_fold_scaled, X_test_fold_scaled, model_fold; gc.collect() 

# --- Walk-Forward Results Evaluation (by Threshold) ---
print("\n--- Aggregated Walk-Forward Validation Results (LightGBM) ---")
if not all_actuals: print("No predictions.")
else:
    print(f"Total OOS predictions: {len(all_actuals)}")
    wf_results_df = pd.DataFrame({'timestamp': all_test_indices_original, 'probability': all_predictions_proba, 'actual': all_actuals}).set_index('timestamp')
    print("\nOverall Performance across Thresholds:"); print("-" * 60)
    for threshold in THRESHOLDS_TO_TEST:
         print(f"Threshold = {threshold:.2f}"); wf_results_df[f'prediction'] = (wf_results_df['probability'] >= threshold).astype(int)
         print(classification_report(wf_results_df['actual'], wf_results_df[f'prediction'], zero_division=0)); print("-" * 60)
    # Plot baseline CM
    baseline_predictions = (wf_results_df['probability'] >= PREDICTION_THRESHOLD).astype(int); cm_wf = confusion_matrix(wf_results_df['actual'], baseline_predictions)
    try: # Plotting CM
        plt.figure(figsize=(8, 6)); plt.imshow(cm_wf, interpolation='nearest', cmap=plt.cm.Blues); plt.title(f'Walk-Forward CM (LGBM, Thresh={PREDICTION_THRESHOLD:.2f})'); plt.colorbar(); unique_targets = np.unique(all_actuals); tick_marks = np.arange(len(unique_targets)); plt.xticks(tick_marks, unique_targets); plt.yticks(tick_marks, unique_targets); plt.xlabel('Predicted label'); plt.ylabel('True label'); thresh = cm_wf.max() / 2.; [plt.text(j, i, f'{cm_wf[i, j]}', ha='center', va='center', color='white' if cm_wf[i, j] > thresh else 'black') for i, j in np.ndindex(cm_wf.shape)]; plt.tight_layout(); plt.savefig(CONFUSION_MATRIX_FILENAME); plt.close()
        print(f"WF confusion matrix saved to {CONFUSION_MATRIX_FILENAME}")
    except Exception as e: print(f"Error saving CM plot: {e}")

# ==============================================================================
#           <<< NEW: FINAL MODEL TRAINING & SHAP ANALYSIS (LGBM) >>>
# ==============================================================================
print("\n--- Training Final Model on ALL Data (LightGBM) ---")

# Prepare final data (X_full, y_full defined above)
scaler_final = StandardScaler() # Use a final scaler
X_full_scaled = scaler_final.fit_transform(X_full.values) # Scale all feature data

# Use average weight or recalculate on full dataset
final_scale_pos_weight = (y_full == 0).sum() / (y_full == 1).sum() if (y_full == 1).sum() > 0 else 1
print(f"Scale pos weight for final model: {final_scale_pos_weight:.2f}")

# Initialize final model (use optimal parameters if tuned, else use base params)
final_model = lgb.LGBMClassifier(
    objective='binary', 
    #eval_metric='logloss', # Note: eval_metric not used without eval_set
    n_estimators=200, learning_rate=0.05, num_leaves=31, 
    scale_pos_weight=final_scale_pos_weight, random_state=42, n_jobs=-1
    # No early stopping when fitting on all data
)
print("Training final LightGBM model...")
final_model.fit(X_full_scaled, y_full) 
print("Final model training complete.")


# --- Feature Importance (From Final LGBM Model) ---
print("\n--- Final LightGBM Feature Importance ---")
try:
    feature_names_list = X_full.columns.tolist() # Get names from DataFrame X_full
    if len(feature_names_list) == len(final_model.feature_importances_):
        feature_importance = pd.DataFrame({'feature': feature_names_list, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
        print("(Top 15):"); print(feature_importance.head(15))
        plt.figure(figsize=(10, 8)); plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
        plt.xlabel("LGBM Feature Importance"); plt.ylabel("Feature"); plt.title("Top 15 Feature Importances (Final LGBM Model)")
        plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(FEATURE_IMPORTANCE_FILENAME); plt.close()
        print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_FILENAME}")
    else: print("Error: Feature name/importance mismatch")
except Exception as e: print(f"Error plotting feature importance: {e}")


# --- <<< NEW: SHAP Value Calculation and Plotting (for LightGBM) >>> ---
print("\n--- Calculating SHAP Values (LightGBM - may take time) ---")
try:
    # SHAP works well with LightGBM using TreeExplainer
    # Need to explain on a reasonably sized sample, e.g., the test split portion
    split_index = int(len(X_full) * (1 - TEST_SET_SIZE)) # Recalculate split point
    X_test_shap = X_full.iloc[split_index:] # Use test portion from original split
    # Use the scaler fitted on the FINAL training data (scaler_final)
    X_test_shap_scaled = scaler_final.transform(X_test_shap.values) 
    
    # Need feature names for SHAP plots
    X_test_shap_df = pd.DataFrame(X_test_shap_scaled, columns=features_to_use)

    print(f"Explaining model on {X_test_shap_scaled.shape[0]} test samples...")
    explainer = shap.TreeExplainer(final_model) 
    shap_values = explainer.shap_values(X_test_shap_scaled) # Calculate SHAP values

    # Check if shap_values is a list (for binary classification, usually returns list [shap_for_class_0, shap_for_class_1])
    shap_values_to_plot = shap_values
    if isinstance(shap_values, list) and len(shap_values) == 2:
         print("Using SHAP values for the positive class (class 1).")
         shap_values_to_plot = shap_values[1] # Explain probability of class 1

    print("Generating SHAP summary plot...")
    # Create summary plot (e.g., beeswarm)
    shap.summary_plot(shap_values_to_plot, X_test_shap_df, show=False) # Use scaled features with names
    plt.title("SHAP Summary Plot (LGBM - Impact on Model Output - Class 1 Prob)")
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PLOT_FILENAME) 
    plt.close()
    print(f"SHAP summary plot saved to {SHAP_SUMMARY_PLOT_FILENAME}")
    
    del shap_values, X_test_shap, X_test_shap_scaled, X_test_shap_df, explainer; gc.collect()

except Exception as e:
    print(f"Error during SHAP analysis: {e}")
    print("Ensure 'shap' library is installed (pip install shap)")


# --- Save Final Artifacts ---
print("\n--- Saving Final Artifacts (LightGBM Model) ---")
try:
    joblib.dump((final_model, scaler_final, features_to_use), MODEL_SAVE_PATH) # Use final scaler
    print(f"Final LightGBM model, scaler, and features saved to {MODEL_SAVE_PATH}")
except Exception as e: print(f"Error saving final artifacts: {e}")

# --- Final Overall Backtest Section is Removed --- 
# Evaluation should rely on the Walk-Forward Results


print("\nWalk-Forward + SHAP Analysis Script Finished.")
