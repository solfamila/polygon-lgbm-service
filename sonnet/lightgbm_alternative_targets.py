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

# --- ML/DL & Utility Imports ---
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# --- Plotting and Saving Imports ---
import matplotlib.pyplot as plt
import joblib 

# --- Technical Analysis Library Import ---
try:
    import pandas_ta as ta; PANDAS_TA_AVAILABLE = True; print("pandas_ta imported.")
except ImportError: PANDAS_TA_AVAILABLE = False; print("Warning: pandas_ta not installed.")

# --- Suppress specific known Pandas FutureWarning ---
import warnings
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.*")

print("\n--- Configuration & Setup ---")

# --- Configuration Loading & Constants ---
load_dotenv(); DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST="localhost"; DB_PORT="5433"; DB_NAME="polygondata"; DB_USER="polygonuser"
TICKER = "TSLA"; 
LOOKAHEAD_PERIOD = 15; TEST_SET_SIZE = 0.2; INITIAL_TRAIN_PCT = 0.60
ROLLING_WINDOW_PCT = 0.10; EARLY_STOPPING_ROUNDS_LGBM = 20 

# --- Target Choice & Related Parameters ---
TARGET_TYPE = 'multiclass'  # <<< MODIFY THIS >>> 'binary', 'regression', 'multiclass'
TARGET_GAIN = 1.0          # Used if TARGET_TYPE=='binary'
PCT_CHANGE_THRESH = 0.005  # Used if TARGET_TYPE=='multiclass'
PREDICTION_THRESHOLD = 0.6 # Baseline threshold for binary/multiclass plots/analysis

# --- Regime Detection Parameters ---
REGIME_VOL_WINDOW = 60 * 8 ; REGIME_LOW_THRESH = 30; REGIME_MED_THRESH = 60   

EXCLUDE_FROM_FEATURES = ['target_binary', 'price_change_pct', 'movement_class', # All possible targets
                         'future_price', 'open', 'high', 'low', 'close', 'volume', 
                         'vwap', 'num_trades', 'hour', 'dayofweek', 
                         'regime'] # Also exclude regime itself from features

# --- Output filenames --- Adjusted based on TARGET_TYPE ---
CONFUSION_MATRIX_FILENAME = f'lgbm_wf_cm_{TARGET_TYPE}_{TICKER.lower()}_thresh{PREDICTION_THRESHOLD:.2f}.png' # Add threshold
# Optional filenames (can uncomment if adding final model train/save back)
# MODEL_SAVE_PATH = f'lgbm_{TARGET_TYPE}_model_final_{TICKER.lower()}.joblib' 
# SCALER_COLS_FILENAME = f'lgbm_{TARGET_TYPE}_scaler_cols_final_{TICKER.lower()}.joblib'
# FEATURE_IMPORTANCE_FILENAME = f'lgbm_final_feat_importance_{TARGET_TYPE}_{TICKER.lower()}.png' 

# --- Database Connection & Data Loading ---
conn = None
df_loaded = pd.DataFrame() # <<< FIX: Initialize df_loaded >>>
 
try:
    print(f"\nConnecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    print("Connection successful."); print(f"Fetching historical data for {TICKER}...")
    query = """SELECT start_time AS time, agg_open AS open, agg_high AS high, agg_low AS low, agg_close AS close, volume, vwap, num_trades FROM stock_aggregates_min WHERE symbol = %(ticker)s AND start_time IS NOT NULL ORDER BY start_time ASC;"""
    df_loaded = pd.read_sql(query, conn, params={'ticker': TICKER}, index_col='time', parse_dates={'time': {'utc': True}})
    print(f"Loaded {len(df_loaded)} historical bars for {TICKER} from DB.")

except psycopg2.Error as e: 
    print(f"Database error during load: {e}")
    # df_loaded remains empty
except Exception as e: 
    print(f"An error occurred during data loading: {e}")
    # df_loaded remains empty
finally:
    if conn: 
        conn.close()
        print("Database connection closed.")

# --- CRITICAL CHECK AFTER LOADING ---        
if df_loaded.empty: 
    print("No data loaded from database. Exiting.")
    exit(1)

# --- Enhanced Feature Engineering Function Definition ---
def create_features(df_input): 
    print("Generating features...")
    df = df_input.copy() # Work on internal copy
    required_cols=['open','high','low','close','volume']; required_ta_cols=['high','low','close','volume']
    if not all(col in df.columns for col in required_cols): print(f"Error: Missing required columns: {required_cols}"); return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
         try: df.index = pd.to_datetime(df.index, utc=True); print("Converted index.")
         except Exception as e: print(f"Error converting index: {e}"); return pd.DataFrame()
    if not df.index.is_monotonic_increasing: print("Warning: Sorting index..."); df.sort_index(inplace=True)

    initial_rows = len(df)
    
    # 1. Basic Features
    df['return'] = df['close'].pct_change()
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['log_volume'] = np.log1p(df['volume'].replace(0, 1)) 
    df['price_vwap_diff'] = df['close'] - df['vwap'] if 'vwap' in df.columns else 0
    df['price_vwap_diff'] = df['price_vwap_diff'].fillna(0) 

    # 2. Moving Averages & Ratios
    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = df['close'].rolling(window=window, min_periods=window).mean()
        df[f'ma_vol_{window}'] = df['volume'].rolling(window=window, min_periods=window).mean()
        df[f'close_ma_{window}_ratio'] = (df['close'] / df[f'ma_{window}']).replace([np.inf, -np.inf], np.nan)
        df[f'volume_ma_{window}_ratio'] = (df['volume'] / df[f'ma_vol_{window}']).replace([np.inf, -np.inf], np.nan)

    # 3. Momentum / ROC
    for window in [1, 5, 10, 20]: df[f'roc_{window}'] = df['close'].pct_change(periods=window)

    # 4. Volatility
    df['volatility_20'] = df['return'].rolling(window=20, min_periods=20).std()
    
    # TA Features 
    ta_cols_to_add = ['ATRr_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'RSI_14']
    for col in ta_cols_to_add: 
        if col not in df.columns: df[col] = np.nan # Initialize columns first

    if PANDAS_TA_AVAILABLE:
        print("Calculating pandas_ta features...")
        try:
            if not all(col in df.columns for col in required_ta_cols): raise ValueError("Missing TA cols")
            df.ta.atr(length=14, append=True); 
            bbands_df = df.ta.bbands(length=20, std=2, append=False); 
            if bbands_df is not None: df.update(bbands_df) # Update safely
            df.ta.rsi(length=14, append=True)
            print("TA features added.")
        except Exception as e: print(f"Warn: TA error: {e}")

    # 5. Volume delta
    df['volume_delta'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan)

    # 6. Time features
    df['hour'] = df.index.hour; df['dayofweek'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0); df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek']/7.0); df['day_cos'] = np.cos(2 * np.pi * df['dayofweek']/7.0)
    
    # Final Feature List & NaN Drop
    # Exclude columns that shouldn't influence NaN drop decisions (like raw OHLCV etc.)
    features_calculated = [col for col in df.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'hour', 'dayofweek']
                             + ['future_price', 'target_binary', 'price_change_pct', 'movement_class'] # Also exclude targets
                             + ['regime']] # Exclude regime if added before this func call
    features_calculated = [col for col in features_calculated if col in df.columns] # Ensure they exist
                             
    print(f"Dropping NaNs based on {len(features_calculated)} calculated features...")
    df.dropna(subset=features_calculated, inplace=True) 
    print(f"Finished features. Rows dropped: {initial_rows - len(df)}/{initial_rows}. Final shape: {df.shape}")
    return df

# --- Regime Detection Function Definition ---
def detect_market_regime(price_series, window=REGIME_VOL_WINDOW, low_thresh=REGIME_LOW_THRESH, med_thresh=REGIME_MED_THRESH):
    print(f"Detecting regimes (window={window}, low<{low_thresh} <med<{med_thresh} <high)...")
    minute_returns = price_series.pct_change(); rolling_std = minute_returns.rolling(window=window,min_periods=window//2).std() 
    annualization_factor = 313.5 ; annualized_vol = rolling_std * annualization_factor * 100 
    regimes = pd.Series(index=price_series.index, dtype=object); regimes.fillna('undefined', inplace=True)
    regimes.loc[annualized_vol <= low_thresh] = 'low_vol'; regimes.loc[(annualized_vol > low_thresh) & (annualized_vol <= med_thresh)] = 'medium_vol'; regimes.loc[annualized_vol > med_thresh] = 'high_vol'
    print("Regime detection complete."); return regimes

# --- Alternative Target Definition Function Definition ---
def add_alternative_targets(df, lookahead=LOOKAHEAD_PERIOD, pct_thresh=PCT_CHANGE_THRESH, gain_thresh=TARGET_GAIN):
    print(f"Adding alternative targets..."); 
    if df.empty: print("Empty df for targets"); return df
    if 'close' not in df.columns: print("No 'close'."); return pd.DataFrame() 
    df_target = df.copy() 
    df_target['future_price'] = df_target['close'].shift(-lookahead) 
    df_target['target_binary'] = ((df_target['future_price'] - df_target['close']) >= gain_thresh).astype(int)
    df_target['price_change_pct'] = df_target['future_price'].sub(df_target['close']).div(df_target['close'].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    conditions = [ df_target['price_change_pct'] <= -pct_thresh, (df_target['price_change_pct'] > -pct_thresh) & (df_target['price_change_pct'] < pct_thresh), df_target['price_change_pct'] >= pct_thresh]; choices = ['DOWN', 'SIDEWAYS', 'UP'] 
    df_target['movement_class'] = np.select(conditions, choices, default='SIDEWAYS')
    target_cols_to_check = ['future_price', 'target_binary', 'price_change_pct', 'movement_class']; initial_len = len(df_target); df_target.dropna(subset=target_cols_to_check, how='any', inplace=True); final_len = len(df_target)
    print(f"Rows after target NaNs drop: {final_len} (removed {initial_len - final_len})")
    if final_len == 0: print("Warning: All rows dropped after target.")
    return df_target


# ==============================================================================
#                   WALK-FORWARD VALIDATION IMPLEMENTATION
# ==============================================================================
print(f"\n--- Starting Walk-Forward Validation (LightGBM - Target: {TARGET_TYPE}) ---")

# 1. Prepare Data
print("Step 1: Preparing full dataset...")
# <<< Corrected CALL ORDER >>>
features_only_df = create_features(df_loaded) # Use the loaded dataframe
if features_only_df.empty: exit("Feature generation failed.")
features_only_df['regime'] = detect_market_regime(features_only_df['close']) 
print("Regime counts in feature set:"); print(features_only_df['regime'].value_counts())
feature_df = add_alternative_targets(features_only_df) # Now add targets
if feature_df.empty: exit("Target generation failed or resulted in empty df."); 

# --- Select Target & Features ---
label_encoder = None; NUM_CLASSES=None # Initialize
if TARGET_TYPE == 'binary': TARGET_COLUMN = 'target_binary'
elif TARGET_TYPE == 'regression': TARGET_COLUMN = 'price_change_pct'
elif TARGET_TYPE == 'multiclass':
    TARGET_COLUMN = 'movement_class'; label_encoder = LabelEncoder()
    if TARGET_COLUMN not in feature_df.columns: exit(f"Target '{TARGET_COLUMN}' not found.")
    feature_df[TARGET_COLUMN] = label_encoder.fit_transform(feature_df[TARGET_COLUMN])
    NUM_CLASSES = len(label_encoder.classes_); print(f"Multiclass Classes: {dict(zip(range(NUM_CLASSES), label_encoder.classes_))}")
else: exit(f"Invalid TARGET_TYPE '{TARGET_TYPE}'")
print(f"Using Target Column: '{TARGET_COLUMN}'")

features_to_use = [col for col in feature_df.columns if col not in EXCLUDE_FROM_FEATURES]
print(f"Using {len(features_to_use)} features.")
X_full = feature_df[features_to_use]; y_full = feature_df[TARGET_COLUMN] 
if len(feature_df) < 100: exit("Not enough data after processing.")

# --- Walk-Forward Loop ---
n_samples = len(X_full); initial_train_size = int(n_samples * INITIAL_TRAIN_PCT)
rolling_window_size = int(n_samples * ROLLING_WINDOW_PCT); 
if rolling_window_size < 1: rolling_window_size = 1 
print(f"WF Params: Samples={n_samples}, InitialTrain={initial_train_size}, PredictWin={rolling_window_size}")
all_test_indices_original=[]; all_actuals=[]; all_predictions_proba = []; all_predictions_reg = [] # Initialize result lists
fold = 0; current_test_start_index = initial_train_size; scaler_wf = StandardScaler() 

while current_test_start_index < n_samples:
    fold += 1; train_indices = np.arange(0, current_test_start_index); test_indices = np.arange(current_test_start_index, min(current_test_start_index + rolling_window_size, n_samples))
    if len(test_indices) == 0: break 
    print(f"\n--- Fold {fold}: Train {len(train_indices)}, Test {len(test_indices)} ---")
    X_train_fold, y_train_fold = X_full.iloc[train_indices].values, y_full.iloc[train_indices].values; X_test_fold, y_test_fold = X_full.iloc[test_indices].values, y_full.iloc[test_indices].values
    original_test_fold_indices = X_full.iloc[test_indices].index; regimes_test_fold = feature_df['regime'].iloc[test_indices].values

    X_train_fold_scaled = scaler_wf.fit_transform(X_train_fold); X_test_fold_scaled = scaler_wf.transform(X_test_fold)

    model_fold = None; fit_params = {}; objective=''; metric='' # Define defaults

    if TARGET_TYPE == 'regression':
        objective='mae'; metric='mae';
        model_fold = lgb.LGBMRegressor(objective=objective, metric=metric, n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)
        fit_params['eval_metric'] = metric
    else: # Classification
        objective = 'binary' if TARGET_TYPE == 'binary' else 'multiclass'; 
        metric = 'logloss' if TARGET_TYPE == 'binary' else 'multi_logloss'; 
        num_class_param = NUM_CLASSES if TARGET_TYPE=='multiclass' else None 
        scale_pos_weight = None
        if TARGET_TYPE == 'binary': scale_pos_weight = (y_train_fold == 0).sum() / (y_train_fold == 1).sum() if (y_train_fold == 1).sum() > 0 else 1; print(f"  Scale pos weight: {scale_pos_weight:.2f}")
        model_fold = lgb.LGBMClassifier( objective=objective, metric=metric, n_estimators=1000, learning_rate=0.05, num_leaves=31, scale_pos_weight=scale_pos_weight, num_class=num_class_param, random_state=42, n_jobs=-1)
        fit_params['eval_metric'] = metric
        
    if model_fold is None: exit("Model init failed.")
    print(f"  Training LGBM fold {fold} (Obj: {objective}, Metric: {metric})..."); eval_set_fold = [(X_test_fold_scaled, y_test_fold)]; callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS_LGBM, verbose=False)]
    model_fold.fit(X_train_fold_scaled, y_train_fold, eval_set=eval_set_fold, callbacks=callbacks, **fit_params)
    
    print(f"  Predicting test fold {fold}..."); 
    if TARGET_TYPE == 'regression': all_predictions_reg.extend(model_fold.predict(X_test_fold_scaled))
    else: all_predictions_proba.append(model_fold.predict_proba(X_test_fold_scaled)) # Store full proba arrays

    all_test_indices_original.extend(original_test_fold_indices); all_actuals.extend(y_test_fold); 
    # Removed regime storing here, get it from merged df later
    current_test_start_index = test_indices[-1] + 1; del X_train_fold, y_train_fold, X_test_fold, y_test_fold, X_train_fold_scaled, X_test_fold_scaled, model_fold; gc.collect() 

# --- Combine and Evaluate Walk-Forward Results ---
print(f"\n--- Aggregated Walk-Forward Results ({TARGET_TYPE}) ---")
if not all_actuals: print("No predictions generated.")
else:
    print(f"Total OOS predictions: {len(all_actuals)}")
    # Process results based on TARGET_TYPE
    if TARGET_TYPE == 'regression':
        wf_results_df = pd.DataFrame({'timestamp': all_test_indices_original, 'prediction': all_predictions_reg, 'actual': all_actuals}).set_index('timestamp')
        # Merge regime info back based on timestamp index
        wf_results_df = wf_results_df.join(feature_df[['regime']], how='left') 
        mae=mean_absolute_error(wf_results_df['actual'],wf_results_df['prediction']); rmse=mean_squared_error(wf_results_df['actual'],wf_results_df['prediction'],squared=False); r2=r2_score(wf_results_df['actual'],wf_results_df['prediction'])
        print("\nOverall Regression Performance:"); print(f"  MAE: {mae:.6f}"); print(f"  RMSE: {rmse:.6f}"); print(f"  R2 Score: {r2:.4f}")
        try: # Plotting Regression
            plt.figure(figsize=(10,6)); plt.scatter(wf_results_df['actual'], wf_results_df['prediction'], c=wf_results_df['regime'].astype('category').cat.codes, cmap='viridis', alpha=0.4) # Color by regime
            plt.plot([wf_results_df['actual'].min(), wf_results_df['actual'].max()], [wf_results_df['actual'].min(), wf_results_df['actual'].max()], 'r--'); plt.xlabel("Actual % Change"); plt.ylabel("Predicted % Change"); plt.title(f"{TICKER} WF Regression"); plt.grid(True); plt.tight_layout(); plt.savefig(f'lgbm_wf_regression_scatter_{TICKER.lower()}.png'); plt.close(); print(f"Regression scatter plot saved.")
        except Exception as e: print(f"Error plotting regression: {e}")
        # Regression analysis per regime
        print("\nRegression Performance BY REGIME:")
        for regime in wf_results_df['regime'].unique():
             print(f"--- Regime: {regime} ---"); regime_data = wf_results_df[wf_results_df['regime']==regime];
             if len(regime_data) < 2: print("  Not enough data for metrics."); continue # R2 needs at least 2 points
             mae_r=mean_absolute_error(regime_data['actual'], regime_data['prediction']); rmse_r=mean_squared_error(regime_data['actual'], regime_data['prediction'], squared=False); r2_r=r2_score(regime_data['actual'], regime_data['prediction'])
             print(f"  Samples: {len(regime_data)}"); print(f"  MAE: {mae_r:.6f}"); print(f"  RMSE: {rmse_r:.6f}"); print(f"  R2 Score: {r2_r:.4f}")
             
    else: # Binary or Multiclass Classification
         pred_class_list = [] # To store final class prediction
         proba_dict = {} # To store probabilities if needed
         if TARGET_TYPE == 'binary':
             proba_class1 = all_predictions_proba # Is already a flat list
             wf_results_df = pd.DataFrame({'timestamp': all_test_indices_original, 'probability': proba_class1, 'actual': all_actuals}).set_index('timestamp')
             wf_results_df = wf_results_df.join(feature_df[['regime']], how='left')
         elif TARGET_TYPE == 'multiclass':
             try: all_predictions_proba_stacked = np.vstack(all_predictions_proba) 
             except ValueError: print("Error stacking probabilities"); all_predictions_proba_stacked=np.array([])
             if all_predictions_proba_stacked.shape[0] != len(all_actuals): print("Probability shape mismatch!"); exit(1)
             pred_class_list = np.argmax(all_predictions_proba_stacked, axis=1)
             # Store individual class probabilities maybe? Requires careful DF construction
             wf_results_df = pd.DataFrame({'timestamp': all_test_indices_original, 'predicted_class': pred_class_list, 'actual': all_actuals}).set_index('timestamp')
             wf_results_df = wf_results_df.join(feature_df[['regime']], how='left')
             
         print("\nOverall Performance across Thresholds/Classes BY REGIME:"); print("=" * 70)
         unique_regimes = wf_results_df['regime'].unique(); 
         for regime in sorted([r for r in unique_regimes if pd.notna(r)]): # Exclude potential NaNs in regime
            print(f"\n--- Regime: {regime} ---"); regime_df = wf_results_df[wf_results_df['regime'] == regime]; print(f"  Samples: {len(regime_df)}")
            if len(regime_df) == 0: continue; 
            print("-" * 60)
            if TARGET_TYPE == 'binary':
                 for threshold in THRESHOLDS_TO_TEST:
                      print(f"    Threshold = {threshold:.2f}")
                      # Use .loc for assignment to avoid warning
                      regime_df.loc[:, f'prediction'] = (regime_df['probability'] >= threshold).astype(int) 
                      try: print(classification_report(regime_df['actual'], regime_df['prediction'], zero_division=0))
                      except Exception as e: print(f"      Report Error: {e}")
                      print("-" * 60)
                 baseline_predictions = (regime_df['probability'] >= PREDICTION_THRESHOLD).astype(int); target_names = ['0', '1']
                 cm_wf = confusion_matrix(regime_df['actual'], baseline_predictions)
                 plot_title=f'WF CM ({regime}, Thresh={PREDICTION_THRESHOLD:.2f})'
                 cm_filename = f'lgbm_wf_cm_{regime}_{TICKER.lower()}_thresh{PREDICTION_THRESHOLD:.2f}.png' # Unique CM per regime

            elif TARGET_TYPE == 'multiclass':
                 print(f"    Threshold = N/A (Multiclass)")
                 target_names = label_encoder.classes_ if label_encoder else ['DOWN', 'SIDEWAYS', 'UP'] # Ensure target names correct
                 print(classification_report(regime_df['actual'], regime_df['predicted_class'], target_names=target_names, zero_division=0))
                 cm_wf = confusion_matrix(regime_df['actual'], regime_df['predicted_class'])
                 plot_title=f'WF CM ({regime}, Multiclass)'
                 cm_filename = f'lgbm_wf_cm_{regime}_{TARGET_TYPE}_{TICKER.lower()}.png' # Unique CM per regime
                 print("-" * 60)

            # Plot Confusion Matrix per Regime
            try: 
                 plt.figure(figsize=(8, 6)); plt.imshow(cm_wf, interpolation='nearest', cmap=plt.cm.Blues); plt.title(plot_title); 
                 plt.colorbar(); tick_marks=np.arange(len(target_names)); plt.xticks(tick_marks, target_names, rotation=45); plt.yticks(tick_marks, target_names); 
                 plt.xlabel('Predicted label'); plt.ylabel('True label'); thresh = cm_wf.max() / 2.; [plt.text(j, i, f'{cm_wf[i, j]}', ha='center', va='center', color='white' if cm_wf[i, j] > thresh else 'black') for i, j in np.ndindex(cm_wf.shape)]; 
                 plt.tight_layout(); plt.savefig(cm_filename); plt.close()
                 print(f"  Confusion matrix saved to {cm_filename}")
            except Exception as e: print(f"  Error saving CM plot: {e}")
            print(f"--- End Regime: {regime} ---\n"); print("=" * 70)

# --- Final Model / SHAP / Backtest Section Removed --- 

print(f"\nWalk-Forward ({TARGET_TYPE} target) Script Finished.")
