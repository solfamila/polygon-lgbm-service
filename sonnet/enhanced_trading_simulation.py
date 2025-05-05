import os
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from lgbm_prediction_service import create_features
import joblib
import itertools
from tqdm import tqdm
import time
import warnings
import seaborn as sns

# Add these imports at the top of the file
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Add this function near the top of your file with other utility functions

def prepare_dataframe_for_excel(df):
    """
    Prepare a DataFrame for Excel export by converting timezone-aware
    datetime columns to timezone-naive.
    """
    df_excel = df.copy()
    
    # Identify datetime columns with timezone info
    datetime_cols = [col for col in df_excel.columns 
                    if pd.api.types.is_datetime64_any_dtype(df_excel[col]) 
                    and df_excel[col].dt.tz is not None]
    
    # Convert timezone-aware columns to timezone-naive
    for col in datetime_cols:
        df_excel[col] = df_excel[col].dt.tz_localize(None)
        
    return df_excel

def fetch_historical_training_data(ticker, months=3):
    """
    Fetch X months of historical minute data for training purposes
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30*months)
    
    # Format dates for the query
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Fetching {months} months of data for {ticker} ({start_date_str} to {end_date_str})...")
    
    query = """
    SELECT start_time AS time, agg_open AS open, agg_high AS high, 
           agg_low AS low, agg_close AS close, volume, vwap
    FROM stock_aggregates_min
    WHERE symbol = %(ticker)s AND start_time BETWEEN %(start)s AND %(end)s
    ORDER BY start_time ASC;
    """
    
    try:
        df = pd.read_sql(query, conn, params={
            'ticker': ticker, 
            'start': start_date,
            'end': end_date
        }, parse_dates=['time'])
        
        if df.empty:
            print(f"No data found for {ticker} in the specified date range.")
            return None
            
        # Check for and handle duplicates
        if df.duplicated(subset=['time']).any():
            print(f"Warning: Duplicate timestamps found in data for {ticker}.")
            df = df.drop_duplicates(subset=['time'], keep='first')
            
        df.set_index('time', inplace=True)
        print(f"Successfully fetched {len(df)} data points.")
        return df
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def create_training_targets(df, lookahead_period=15, target_gain=1.0):
    """
    Create binary target labels from historical data
    based on price movements after lookahead_period
    """
    df_with_target = df.copy()
    
    # Calculate future returns for the lookahead period
    future_returns = df['close'].pct_change(periods=lookahead_period).shift(-lookahead_period) * 100
    
    # Create binary targets
    binary_target = (future_returns >= target_gain).astype(int)
    
    df_with_target['target'] = binary_target
    
    # Remove rows where we don't have targets (at the end of the dataframe)
    df_with_target = df_with_target.dropna(subset=['target'])
    
    # Show class distribution
    target_distribution = df_with_target['target'].value_counts(normalize=True) * 100
    print(f"Target distribution: {target_gain}% gain opportunities: {target_distribution[1]:.2f}%")
    
    return df_with_target


def train_model_on_historical_data(ticker, months=3, lookahead_period=15, target_gain=1.0):
    """
    Complete pipeline to train a new model on recent historical data
    """
    # 1. Fetch historical data
    historical_data = fetch_historical_training_data(ticker, months)
    if historical_data is None or historical_data.empty:
        return None
    
    # 2. Generate features
    print("Generating features...")
    feature_df = create_features(historical_data)
    
    # 3. Create target labels
    print("Creating target labels...")
    labeled_df = create_training_targets(feature_df, lookahead_period, target_gain)
    
    # 4. Prepare training data
    print("Preparing training data...")
    X = labeled_df.drop(['target', 'open', 'high', 'low', 'close', 'volume', 'vwap'], axis=1)
    y = labeled_df['target']
    
    # Check for NaNs and handle them
    nan_columns = X.columns[X.isna().any()].tolist()
    if nan_columns:
        print(f"Warning: NaN values found in columns: {nan_columns}")
        X = X.fillna(0)
    
    # 5. Train-test split (time-based)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 6. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Calculate class weights
    if (y_train == 1).sum() > 0:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        scale_pos_weight = 1.0
    
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 8. Train LightGBM model
    print("Training new LightGBM model...")
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    # Early stopping callback
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
    
    # Fit the model
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric='binary_logloss',
        callbacks=callbacks
    )
    
    # Add to train_model_on_historical_data function, before saving the model
    # Check if model is producing useful predictions
    print("Validating model prediction distribution...")
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    if test_probs.max() < 0.1:
        print("Warning: Model is producing very low confidence scores")
        print(f"Max confidence: {test_probs.max():.4f}, Mean confidence: {test_probs.mean():.4f}")
        print("Adjusting model parameters to produce more useful predictions...")
        
        # Train a more aggressive version with different parameters
        model_aggressive = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=5,  # Set explicit depth
            min_child_samples=10,  # Lower to allow more detailed patterns
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight * 0.5,  # Lower to give more weight to positives
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit without early stopping to ensure more iterations
        model_aggressive.fit(X_train_scaled, y_train)
        
        # Check if the aggressive model produces better distributed predictions
        aggressive_probs = model_aggressive.predict_proba(X_test_scaled)[:, 1]
        
        if aggressive_probs.max() > test_probs.max() * 2:
            print("Aggressive model produces better distributed predictions")
            print(f"New max confidence: {aggressive_probs.max():.4f}, New mean: {aggressive_probs.mean():.4f}")
            model = model_aggressive
    
    # 9. Evaluate the model
    print("Evaluating model...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    # 10. Save the model and necessary artifacts
    print("Saving model and artifacts...")
    model_path = f'lgbm_final_model_enhanced_{ticker.lower()}.joblib'
    feature_columns = X.columns.tolist()
    
    joblib.dump((model, scaler, feature_columns), model_path)
    print(f"Model saved to {model_path}")
    
    # Return the trained model and artifacts
    return model, scaler, feature_columns


def learn_from_past_trading_mistakes(ticker, months=3, iterations=3):
    """
    Train-Test-Learn cycle:
    1. Train model on historical data
    2. Backtest on out-of-sample data
    3. Analyze mistakes and retrain
    4. Repeat
    """
    print(f"Starting learning process for {ticker} with {iterations} iterations...")
    
    # Initial training
    result = train_model_on_historical_data(ticker, months)
    if result is None:
        print(f"Could not train initial model for {ticker}. Please check if historical data is available.")
        return None
        
    model, scaler, features = result
    
    # Iterative learning process
    for iteration in range(1, iterations):
        print(f"\n--- Iteration {iteration} of {iterations-1} ---")
        
        # 1. Backtest the model on recent data
        historical_data = fetch_historical_training_data(ticker, months=1)  # Use 1 month for backtesting
        if historical_data is None or historical_data.empty:
            print(f"Warning: Could not fetch backtesting data for iteration {iteration}")
            continue
            
        # 2. Generate features
        print("Generating features for backtesting...")
        feature_df = create_features(historical_data)
        
        # 3. Make predictions and find mistakes
        print("Making predictions to identify mistakes...")
        X_backtest = feature_df.drop(['open', 'high', 'low', 'close', 'volume', 'vwap'], axis=1)
        
        # Handle NaNs
        X_backtest = X_backtest.fillna(0)
        
        # Make sure we only use columns the model was trained on
        missing_cols = [col for col in features if col not in X_backtest.columns]
        if missing_cols:
            print(f"Warning: Missing columns in backtest data: {missing_cols}")
            for col in missing_cols:
                X_backtest[col] = 0  # Add missing columns with default values
        
        X_backtest = X_backtest[features]  # Only use features the model knows
        
        # Scale the data
        X_backtest_scaled = scaler.transform(X_backtest)
        
        # Predict
        y_pred_proba = model.predict_proba(X_backtest_scaled)[:, 1]
        
        # Create binary prediction with optimized threshold
        threshold = 0.57  # Using the default confidence threshold from parameters
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 4. Create actual targets for comparison
        print("Creating targets to find mistakes...")
        future_returns = feature_df['close'].pct_change(periods=15).shift(-15) * 100
        y_true = (future_returns >= 1.0).astype(int)
        
        # Remove NaNs from targets and align predictions
        mask = ~y_true.isna()
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # If we don't have enough data for comparison, skip this iteration
        if len(y_true) < 100:
            print(f"Warning: Not enough data points after removing NaNs: {len(y_true)}")
            continue
        
        # 5. Find where the model made mistakes
        print("Identifying mistake periods...")
        mistakes = y_true != y_pred
        mistake_days = feature_df.index[mask][mistakes].tolist()
        
        if not mistake_days:
            print("No mistakes found in this iteration!")
            continue
            
        print(f"Found {len(mistake_days)} periods with mistakes")
        
        # 6. Fetch more data around the mistakes
        print("Fetching additional data for mistake periods...")
        mistake_data = fetch_data_for_mistake_periods(ticker, mistake_days)
        
        if mistake_data is None or mistake_data.empty:
            print(f"Warning: Could not fetch data for mistake periods in iteration {iteration}")
            continue
            
        # 7. Generate features and targets for the mistake data
        print("Generating features for mistake data...")
        mistake_feature_df = create_features(mistake_data)
        
        # 8. Create targets
        print("Creating targets for mistake data...")
        mistake_labeled_df = create_training_targets(mistake_feature_df)
        
        # 9. Prepare the mistake data for training
        print("Preparing mistake data for training...")
        X_mistake = mistake_labeled_df.drop(['target', 'open', 'high', 'low', 'close', 'volume', 'vwap'], axis=1)
        y_mistake = mistake_labeled_df['target']
        
        # Handle NaNs
        X_mistake = X_mistake.fillna(0)
        
        # Ensure we have only the columns the model expects
        for col in features:
            if col not in X_mistake.columns:
                X_mistake[col] = 0
        
        X_mistake = X_mistake[features]
        
        # 10. Scale the data
        X_mistake_scaled = scaler.transform(X_mistake)
        
        # 11. Update the model with the mistake data
        print("Retraining model with mistake data...")
        # Calculate class weights for imbalanced data
        if (y_mistake == 1).sum() > 0:
            scale_pos_weight = (y_mistake == 0).sum() / (y_mistake == 1).sum()
        else:
            scale_pos_weight = 1.0
            
        print(f"Using scale_pos_weight for retraining: {scale_pos_weight:.2f}")
        
        # Update model parameters
        model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Retrain
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
        model.fit(
            X_mistake_scaled, y_mistake,
            eval_set=[(X_mistake_scaled, y_mistake)],
            eval_metric='binary_logloss',
            callbacks=callbacks
        )
        
        print(f"Model retrained in iteration {iteration}")
    
    print("\nIterative learning process completed.")
    return model, scaler, features

def fetch_data_for_mistake_periods(ticker, mistake_days, buffer_days=3):
    """
    Fetch data for periods where the model made mistakes,
    plus a buffer of days before/after
    """
    all_data = []
    
    for day in mistake_days:
        # Convert day to datetime if it's not already
        if isinstance(day, str):
            day = pd.Timestamp(day)
        
        # Add buffer before and after
        start_date = day - timedelta(days=buffer_days)
        end_date = day + timedelta(days=buffer_days)
        
        query = """
        SELECT start_time AS time, agg_open AS open, agg_high AS high, 
               agg_low AS low, agg_close AS close, volume, vwap
        FROM stock_aggregates_min
        WHERE symbol = %(ticker)s AND start_time BETWEEN %(start)s AND %(end)s
        ORDER BY start_time ASC;
        """
        
        try:
            df = pd.read_sql(query, conn, params={
                'ticker': ticker, 
                'start': start_date,
                'end': end_date
            }, parse_dates=['time'])
            
            if not df.empty:
                df.set_index('time', inplace=True)
                all_data.append(df)
        except Exception as e:
            print(f"Error fetching data for mistake period {day}: {e}")
            continue
    
    if not all_data:
        return None
    
    # Combine all fetched data
    combined_data = pd.concat(all_data)
    
    # Remove duplicates that might occur in overlapping periods
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
    
    # Sort by time
    combined_data.sort_index(inplace=True)
    
    print(f"Collected {len(combined_data)} data points from mistake periods.")
    return combined_data


# Add this function near other utility functions at the top of file

def get_best_model_path(ticker):
    """
    Gets the best available model path for a ticker.
    Prioritizes calibrated models over standard models if they exist.
    
    Args:
        ticker (str): The ticker symbol
    
    Returns:
        str: Path to the best available model
    """
    ticker = ticker.lower()
    calibrated_model_path = f'lgbm_calibrated_model_{ticker}.joblib'
    standard_model_path = f'lgbm_final_model_enhanced_{ticker}.joblib'
    
    if os.path.exists(calibrated_model_path):
        print(f"Using calibrated model for {ticker}")
        return calibrated_model_path
    elif os.path.exists(standard_model_path):
        print(f"Using standard model for {ticker}")
        return standard_model_path
    else:
        print(f"No model found for {ticker}")
        return None


# Load environment variables
load_dotenv()
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "polygondata"
DB_USER = "polygonuser"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Default parameters (will be used for parameter optimization)
DEFAULT_PARAMS = {
    'ticker': 'TSLA',
    'confidence_threshold': 0.57,
    'target_profit_pct': 0.01,
    'stop_loss_pct': 0.01,
    'max_shares_per_trade': 50,
    'capital': 10000.0,
    'commission_per_share': 0.005
}

# Temporarily adjust threshold for testing purposes
DEFAULT_PARAMS['confidence_threshold'] = 0.05

# Multi-symbol testing list
TICKERS_TO_TEST = ["TSLA", "AAPL", "MSFT", "AMZN", "NVDA"]

# Connect to DB
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
conn.autocommit = True

# Market regime detection
def detect_market_regime(price_series, window=60*8, low_thresh=30, med_thresh=60):
    """
    Detects market regimes based on volatility
    """
    minute_returns = price_series.pct_change()
    rolling_std = minute_returns.rolling(window=window, min_periods=window//2).std()
    annualized_vol = rolling_std * np.sqrt(252 * 6.5 * 60) * 100  # Annualized volatility in percent
    
    # Define regimes
    regimes = pd.Series(index=price_series.index, dtype=object)
    regimes[annualized_vol <= low_thresh] = 'LOW_VOL'
    regimes[(annualized_vol > low_thresh) & (annualized_vol <= med_thresh)] = 'MED_VOL'
    regimes[annualized_vol > med_thresh] = 'HIGH_VOL'
    regimes.fillna('UNDEFINED', inplace=True)
    
    return regimes

# Fetch price data for a specific date
def fetch_data(ticker, date):
    query = """
    SELECT start_time AS time, agg_open AS open, agg_high AS high, agg_low AS low, 
           agg_close AS close, volume, vwap
    FROM stock_aggregates_min
    WHERE symbol = %(ticker)s AND start_time BETWEEN %(start)s AND %(end)s
    ORDER BY start_time ASC;
    """
    start = pd.Timestamp(date).replace(hour=0, minute=0, second=0, tzinfo=pd.Timestamp.now().tz)
    end = start + pd.Timedelta(days=1)
    df = pd.read_sql(query, conn, params={'ticker': ticker, 'start': start, 'end': end}, parse_dates=['time'])
    
    # Check for and handle duplicates in raw data
    if df.duplicated(subset=['time']).any():
        print(f"Warning: Duplicate timestamps found in raw data for {ticker} on {date}.")
        df = df.drop_duplicates(subset=['time'], keep='first')
        
    df.set_index('time', inplace=True)
    return df

# Simulate trades for a single day
def simulate_single_day(ticker, date, params=None, fetch_missing=False, diagnostic_mode=False, custom_model_path=None):
    if params is None:
        params = DEFAULT_PARAMS.copy()  # Make sure to copy
    
    # Load model artifacts - allow custom model path
    if custom_model_path and os.path.exists(custom_model_path):
        model_path = custom_model_path
        print(f"Using custom model: {model_path}")
    else:
        model_path = f'lgbm_final_model_enhanced_{ticker.lower()}.joblib'
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Skipping {ticker} on {date}.")
        return None, None, None
    
    try:
        model_data = joblib.load(model_path)
        
        # Check if this is a calibrated model package
        if isinstance(model_data, dict) and 'calibration_model' in model_data:
            print(f"Using calibrated model (method: {model_data['calibration_method']}, date: {model_data['calibration_date']})")
            model = model_data['original_model']
            scaler = model_data['scaler']
            features_to_use = model_data['features']
            calibrated_model = True
            calibration_package = model_data
            # Use the optimal threshold if available
            if 'optimal_threshold' in model_data:
                calibrated_threshold = model_data['optimal_threshold']
                print(f"Calibrated threshold available: {calibrated_threshold:.4f}")

                # Allow manual override if provided in params explicitly
                if 'confidence_threshold' not in params:
                    params['confidence_threshold'] = calibrated_threshold
                    print(f"Using calibrated threshold: {calibrated_threshold:.4f}")
                else:
                    print(f"Using manually overridden threshold: {params['confidence_threshold']:.4f}")
        else:
            # Standard model format (model, scaler, features)
            model, scaler, features_to_use = model_data
            calibrated_model = False
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return None, None, None

    # Extract parameters AFTER model loading to use updated threshold
    confidence_threshold = params.get('confidence_threshold', 0.57)
    target_profit_pct = params.get('target_profit_pct', 0.01)
    stop_loss_pct = params.get('stop_loss_pct', 0.01)
    max_shares_per_trade = params.get('max_shares_per_trade', 50)
    initial_capital = params.get('capital', 10000.0)
    commission_per_share = params.get('commission_per_share', 0.005)
    

    # Fetch the day's data
    df_raw = fetch_data(ticker, date)
    if df_raw.empty:
        print(f"No data available for {ticker} on {date}")
        
        # Try to fetch missing data from Polygon if option is enabled
        if fetch_missing:
            try:
                # Attempt to fetch missing data
                fetch_success = fetch_missing_data(ticker, date)
                if fetch_success:
                    print(f"Successfully fetched data for {ticker} on {date}. Retrying simulation...")
                    # Try again with the newly fetched data
                    df_raw = fetch_data(ticker, date)
                    if df_raw.empty:
                        print(f"Still no data available after fetching from Polygon API")
                        return None, None, None
                else:
                    print(f"Failed to fetch data from Polygon API")
                    return None, None, None
            except Exception as e:
                print(f"Error while trying to fetch missing data: {e}")
                return None, None, None
        else:
            return None, None, None
    
    # Generate features for model prediction
    print("Generating features...")
    df_features = create_features(df_raw.copy())
    
    # Ensure we have enough data after feature generation
    if len(df_features) < 60:  # Arbitrary minimum number of bars
        print(f"Not enough data for {ticker} on {date} after feature generation")
        return None, None, None
    
    # Create a copy of the dataframe for model features
    df_features_model = df_features.copy()
    
    # Check for NaN values after feature generation
    nan_count = df_features_model.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found after feature generation.")
        df_features_model = df_features_model.dropna()
        print(f"Dropped rows with NaN values. Shape after drop: {df_features_model.shape}")
        
        if df_features_model.empty:
            print("No valid data left after dropping NaN values")
            return None, None, None
    
    print(f"Finished feature generation. Shape before NaN check: {df_features.shape}")
    
    # Save close prices for trading simulation
    close_prices = df_features['close']
    
    # Now it's safe to assign
    df_features_model['close'] = close_prices
    df_features_model['high'] = df_features['high']
    df_features_model['low'] = df_features['low']
    
    # Add market regime detection
    df_features_model['regime'] = detect_market_regime(df_features_model['close'])

    # After adding regime detection but before scaling
    if 'regime' in features_to_use:
        # Convert string regime values to numeric
        regime_mapping = {
            'LOW_VOL': 0,
            'MED_VOL': 1, 
            'HIGH_VOL': 2,
            'UNDEFINED': -1
        }
        df_features_model['regime'] = df_features_model['regime'].map(regime_mapping)

    # Ensure the features expected by the model exist in the dataframe
    missing_features = [f for f in features_to_use if f not in df_features_model.columns]
    if missing_features:
        print(f"Error: Missing expected features in dataframe: {missing_features}")
        # Attempt to re-generate features if missing, might indicate an issue upstream
        # For now, return None as the state is inconsistent
        return None, None, None

    # Select only the features the model was trained on for scaling and prediction
    # This ensures we only use the columns the model expects
    features_for_prediction = df_features_model[features_to_use].copy()

    # Updated Debug code to show alignment just before scaling
    print(f"Model expects {len(features_to_use)} features.")
    # This print statement now shows the columns actually being used for prediction
    print(f"Features selected for scaling/prediction ({features_for_prediction.shape[1]}): {features_for_prediction.columns.tolist()}")

    # Scale the selected features
    df_features_scaled = scaler.transform(features_for_prediction)

    # Make predictions using the scaled features
    if calibrated_model:
        # Import the prediction function
        from model_calibration import predict_with_calibrated_model
        predictions = predict_with_calibrated_model(calibration_package, df_features_scaled)
    else:
        predictions = model.predict_proba(df_features_scaled)[:, 1]  # Get probability of positive class

    # Add predictions back to the main df for use in simulation logic
    df_features_model['predicted_probability'] = predictions

    # Diagnostic print statements
    exceeding_threshold = (predictions > confidence_threshold).sum()
    print(f"Predictions exceeding threshold ({confidence_threshold:.4f}): {exceeding_threshold}/{len(predictions)}")
    print(f"Prediction stats - Min: {predictions.min():.4f}, Mean: {predictions.mean():.4f}, Max: {predictions.max():.4f}")

    # Early return if no predictions exceed threshold
    if exceeding_threshold == 0:
        print("No predictions exceed the threshold - no trades will be executed.")
        return None, None, None

    # Trading simulation variables
    capital = initial_capital
    shares_owned = 0
    entry_price = 0
    entry_time = None
    trade_count = 0
    trades = []
    equity_curve = pd.Series(index=df_features_model.index, dtype=float)
    equity_curve.iloc[0] = capital
    
    # Trade visualization data
    trade_entries = []
    trade_exits = []
    trade_types = []
    
    # Trading loop
    for idx, row in df_features_model.iterrows():
        current_time = idx
        current_price = row['close']
        confidence = row['predicted_probability']
        
        # Update equity curve
        equity_curve[idx] = capital + (shares_owned * current_price)
        
        # Execute entry logic if we don't have a position
        if shares_owned == 0:
            # Entry logic based on prediction
            if confidence > confidence_threshold:
                # Calculate position size based on capital
                affordable_shares = min(max_shares_per_trade, int(capital / current_price))
                
                # Check for minimum position size
                if affordable_shares < 1:
                    continue
                
                # Adjust for regime - be more conservative in high volatility

                if row['regime'] == 0:  # LOW_VOL
                    position_size_factor = 1.0  # Full size in low vol
                elif row['regime'] == 1:  # MED_VOL
                    position_size_factor = 0.8  # 80% size in medium vol
                else:  # HIGH_VOL or UNDEFINED
                    position_size_factor = 0.5  # 50% size in high vol
                
                # Apply position sizing
                shares_to_buy = max(1, int(affordable_shares * position_size_factor))
                
                # Execute buy
                cost = shares_to_buy * current_price
                commission = shares_to_buy * commission_per_share
                capital -= (cost + commission)
                shares_owned = shares_to_buy
                entry_price = current_price
                entry_time = current_time
                
                # Record entry for visualization
                trade_entries.append((current_time, current_price))
        
        # Execute exit logic if we have a position
        elif shares_owned > 0:
            # Calculate P&L
            unrealized_pl = shares_owned * (current_price - entry_price)
            unrealized_pl_pct = (current_price - entry_price) / entry_price
            
            exit_reason = None
            
            # Target profit exit
            if unrealized_pl_pct >= target_profit_pct:
                exit_reason = 'TARGET'
            
            # Stop loss exit
            elif unrealized_pl_pct <= -stop_loss_pct:
                exit_reason = 'STOP'
            
            # Confidence drop exit
            elif confidence < 0.5:  # Lower threshold for exit
                exit_reason = 'CONFIDENCE'
            
            # End of day exit
            elif current_time.hour >= 15 and current_time.minute >= 45:
                exit_reason = 'EOD'
            
            # Execute exit if we have a reason
            if exit_reason:
                # Execute sell
                proceeds = shares_owned * current_price
                commission = shares_owned * commission_per_share
                capital += (proceeds - commission)
                
                # Calculate profit/loss
                profit_loss = proceeds - (shares_owned * entry_price) - (2 * shares_owned * commission_per_share)
                
                # Record trade
                trade_data = {
                    'date': current_time.date(),
                    'ticker': ticker,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares_owned,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss / (shares_owned * entry_price) * 100,
                    'exit_reason': exit_reason,
                    'confidence': confidence,
                    'regime': row['regime']
                }
                trades.append(trade_data)
                
                # Determine trade type for visualization
                trade_type = 'win' if profit_loss > 0 else 'loss'
                
                # Record exit for visualization
                trade_exits.append((current_time, current_price))
                trade_types.append(trade_type)
                
                # Reset position
                shares_owned = 0
                entry_price = 0
                entry_time = None
                trade_count += 1
    
    # Force exit any remaining positions at end of day
    if shares_owned > 0:
        last_time = df_features_model.index[-1]
        last_price = df_features_model['close'].iloc[-1]
        
        # Execute sell
        proceeds = shares_owned * last_price
        commission = shares_owned * commission_per_share
        capital += (proceeds - commission)
        
        # Calculate profit/loss
        profit_loss = proceeds - (shares_owned * entry_price) - (2 * shares_owned * commission_per_share)
        
        # Record trade
        trade_data = {
            'date': last_time.date(),
            'ticker': ticker,
            'entry_time': entry_time,
            'exit_time': last_time,
            'entry_price': entry_price,
            'exit_price': last_price,
            'shares': shares_owned,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss / (shares_owned * entry_price) * 100,
            'exit_reason': 'CLOSE',
            'confidence': df_features_model['predicted_probability'].iloc[-1],
            'regime': df_features_model['regime'].iloc[-1]
        }
        trades.append(trade_data)
        
        # Determine trade type for visualization
        trade_type = 'win' if profit_loss > 0 else 'loss'
        
        # Record exit for visualization
        trade_exits.append((last_time, last_price))
        trade_types.append(trade_type)
        
        # Reset position
        trade_count += 1
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Calculate summary metrics
    metrics = {}
    if not trades_df.empty:
        total_pnl = trades_df['profit_loss'].sum()
        win_count = (trades_df['profit_loss'] > 0).sum()
        loss_count = (trades_df['profit_loss'] <= 0).sum()
        win_rate = win_count / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        # Calculate risk metrics
        if loss_count > 0:
            avg_win = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].mean() if win_count > 0 else 0
            avg_loss = trades_df[trades_df['profit_loss'] <= 0]['profit_loss'].mean() if loss_count > 0 else 0
            risk_reward = abs(avg_win / avg_loss) if avg_loss < 0 else 0
            profit_factor = abs(trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum() / 
                            trades_df[trades_df['profit_loss'] <= 0]['profit_loss'].sum()) if trades_df[trades_df['profit_loss'] <= 0]['profit_loss'].sum() < 0 else 0
        else:
            avg_win = trades_df['profit_loss'].mean() if win_count > 0 else 0
            avg_loss = 0
            risk_reward = float('inf')
            profit_factor = float('inf')
        
        # Calculate return metrics
        return_pct = (capital - initial_capital) / initial_capital * 100
        
        # Calculate Sharpe ratio (basic version)
        if equity_curve.std() != 0:
            returns = equity_curve.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        metrics = {
            'date': date.date(),
            'ticker': ticker,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_pnl': total_pnl,
            'return_pct': return_pct,
            'trade_count': trade_count,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward': risk_reward,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        }
    else:
        # Diagnostic mode for empty trades
        if diagnostic_mode and trades_df.empty:
            print(f"\n--- Diagnostic Information for {ticker} on {date.date()} ---")
            
            # Confidence score analysis
            confidence_scores = df_features_model['predicted_probability'].values
            print(f"Confidence score statistics:")
            print(f"  Mean: {confidence_scores.mean():.4f}")
            print(f"  Max: {confidence_scores.max():.4f}")
            print(f"  Min: {confidence_scores.min():.4f}")
            print(f"  Values > 0.5: {(confidence_scores > 0.5).sum()} ({(confidence_scores > 0.5).sum()/len(confidence_scores)*100:.2f}%)")
            print(f"  Values > {confidence_threshold}: {(confidence_scores > confidence_threshold).sum()} ({(confidence_scores > confidence_threshold).sum()/len(confidence_scores)*100:.2f}%)")
            
            # Create diagnostic visualization
            plt.figure(figsize=(14, 10))
            
            # 1. Price chart
            plt.subplot(3, 1, 1)
            plt.plot(close_prices.index, close_prices, 'k-', label='Price')
            plt.title(f'{ticker} Price - {date.date()}')
            plt.ylabel('Price ($)')
            plt.grid(True, alpha=0.3)
            
            # 2. Confidence scores
            plt.subplot(3, 1, 2)
            plt.plot(df_features_model.index, df_features_model['predicted_probability'], 'b-')
            plt.axhline(y=confidence_threshold, color='r', linestyle='--', label=f'Threshold ({confidence_threshold})')
            plt.axhline(y=0.5, color='g', linestyle=':', label='0.5 Level')
            plt.title('Model Confidence Scores')
            plt.ylabel('Confidence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. Regime analysis
            plt.subplot(3, 1, 3)
            regimes = df_features_model['regime'].value_counts()
            plt.bar(regimes.index, regimes.values)
            plt.title('Market Regime Distribution')
            plt.ylabel('Number of Periods')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            diagnostic_filename = f"{ticker}_diagnostic_{date.date()}.png"
            plt.savefig(diagnostic_filename)
            plt.close()
            print(f"Diagnostic visualization saved to {diagnostic_filename}")
            
            # Offer to try a lower threshold
            try_lower = input("\nNo trades were executed. Would you like to try with a lower confidence threshold? (y/n): ").lower() == 'y'
            if try_lower:
                new_threshold = float(input("Enter new confidence threshold (0.5-0.57): ") or "0.52")
                # Make sure threshold is reasonable
                new_threshold = max(0.5, min(confidence_threshold, new_threshold))
                print(f"Rerunning simulation with confidence threshold = {new_threshold}")
                
                # Create new params with lower threshold
                new_params = params.copy()
                new_params['confidence_threshold'] = new_threshold
                
                # Recursively call simulate_single_day with new threshold
                return simulate_single_day(ticker, date, new_params, fetch_missing)
        
        # Basic metrics if no trades were executed
        metrics = {
            'date': date.date(),
            'ticker': ticker,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_pnl': 0,
            'return_pct': 0,
            'trade_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'risk_reward': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0
        }
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Prepare visualization data
    viz_data = {
        'price_data': close_prices,
        'equity_curve': equity_curve,
        'trade_entries': trade_entries,
        'trade_exits': trade_exits,
        'trade_types': trade_types,
        'predicted_probability': df_features_model['predicted_probability'] if 'predicted_probability' in df_features_model.columns else None,
        'regime': df_features_model['regime']
    }
    
    # Display summary if trades were executed
    if not trades_df.empty:
        print(f"\n--- Trading Summary for {ticker} on {date.date()} ---")
        print(f"Total Trades: {trade_count}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"Return: {metrics['return_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Price chart with entries and exits
        plt.subplot(2, 1, 1)
        plt.plot(close_prices.index, close_prices, 'k-', label='Price')
        
        for i in range(len(trade_entries)):
            entry_time, entry_price = trade_entries[i]
            exit_time, exit_price = trade_exits[i]
            trade_type = trade_types[i]
            
            if trade_type == 'win':
                plt.plot([entry_time, exit_time], [entry_price, exit_price], 'g-', alpha=0.7)
                plt.scatter(entry_time, entry_price, c='blue', marker='^', s=100)
                plt.scatter(exit_time, exit_price, c='green', marker='v', s=100)
            else:
                plt.plot([entry_time, exit_time], [entry_price, exit_price], 'r-', alpha=0.7)
                plt.scatter(entry_time, entry_price, c='blue', marker='^', s=100)
                plt.scatter(exit_time, exit_price, c='red', marker='v', s=100)
        
        plt.title(f'{ticker} Price with Trades - {date.date()}')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        
        # Equity curve
        plt.subplot(2, 1, 2)
        plt.plot(equity_curve.index, equity_curve, 'b-')
        plt.title('Equity Curve')
        plt.ylabel('Capital ($)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{ticker}_simulation_{date.date()}.png")
        plt.close()
        print(f"Visualization saved to {ticker}_simulation_{date.date()}.png")
    else:
        print(f"No trades executed for {ticker} on {date.date()}")
    
    return trades_df, metrics_df, viz_data

# Function to fetch missing data from Polygon API and save to database
def fetch_missing_data(ticker, date):
    """
    Fetch missing data for a ticker on a specific date from Polygon API
    and save it to the database
    """
    try:
        print(f"Fetching data for {ticker} on {date.date()} from Polygon API...")
        
        # Format date strings for API request
        date_str = date.strftime('%Y-%m-%d')
        
        # Fetch from Polygon API
        aggs_data = fetch_aggregates(
            ticker=ticker,
            start_date=date_str,
            end_date=date_str,
            timespan='minute',
            api_key=POLYGON_API_KEY
        )
        
        if not aggs_data or len(aggs_data) == 0:
            print(f"No data returned from Polygon API for {ticker} on {date.date()}")
            return False
            
        # Save the data to the database
        print(f"Processing and saving {len(aggs_data)} records to database...")
        
        # Create a cursor and save the data
        cursor = conn.cursor()
        
        # Insert data into table
        for agg in aggs_data:
            # Convert timestamp to proper format
            timestamp = pd.Timestamp(agg['timestamp'], unit='ms', tz='UTC')
            
            # Insert into database
            query = """
            INSERT INTO stock_aggregates_min 
            (symbol, start_time, agg_open, agg_high, agg_low, agg_close, volume, vwap)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, start_time) DO UPDATE
            SET agg_open = EXCLUDED.agg_open,
                agg_high = EXCLUDED.agg_high,
                agg_low = EXCLUDED.agg_low,
                agg_close = EXCLUDED.agg_close,
                volume = EXCLUDED.volume,
                vwap = EXCLUDED.vwap;
            """
            
            cursor.execute(query, (
                ticker,
                timestamp,
                agg['open'],
                agg['high'],
                agg['low'],
                agg['close'],
                agg['volume'],
                agg.get('vwap', 0)
            ))
        
        conn.commit()
        cursor.close()
        
        print(f"Successfully fetched and saved data for {ticker} on {date.date()}")
        return True
            
    except Exception as e:
        print(f"Error fetching data for {ticker} on {date.date()}: {e}")
        return False


# Function to run simulations across multiple dates
def multi_day_simulation(ticker, date_list, params=None):
    all_trades = []
    all_metrics = []
    
    for date in tqdm(date_list, desc=f"Simulating {ticker}"):
        try:
            trades_df, metrics_df, _ = simulate_single_day(ticker, date, params)
            if trades_df is not None and not trades_df.empty:
                all_trades.append(trades_df)
            if metrics_df is not None and not metrics_df.empty:
                all_metrics.append(metrics_df)
        except Exception as e:
            print(f"Error processing {ticker} on {date}: {e}")
            continue
    
    # Combine results
    combined_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
    combined_metrics = pd.concat(all_metrics) if all_metrics else pd.DataFrame()
    
    return combined_trades, combined_metrics

# Function to perform parameter optimization
def optimize_parameters(ticker, test_date, param_grid):
    """
    Test multiple parameter combinations to find optimal settings
    """
    results = []
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    for combo in tqdm(param_combinations, desc="Testing parameters"):
        # Create parameter dictionary
        params = DEFAULT_PARAMS.copy()
        for i, param_name in enumerate(param_names):
            params[param_name] = combo[i]
        
        # Run simulation with these parameters
        _, metrics_df, _ = simulate_single_day(ticker, test_date, params)
        
        if metrics_df is not None and not metrics_df.empty:
            # Add parameter values to metrics
            for i, param_name in enumerate(param_names):
                metrics_df[param_name] = combo[i]
            
            results.append(metrics_df)
    
    # Combine results
    results_df = pd.concat(results) if results else pd.DataFrame()
    return results_df

# Function to analyze trades by various factors
def analyze_trading_patterns(trades_df):
    """
    Analyze trading patterns by time of day, confidence level, etc.
    """
    analysis = {}
    
    if trades_df.empty:
        return pd.DataFrame()
    
    # Analysis by hour
    trades_df['entry_hour'] = trades_df['entry_time'].dt.hour
    hourly_performance = trades_df.groupby('entry_hour')['profit_loss'].agg(['mean', 'sum', 'count'])
    hourly_performance['win_rate'] = trades_df.groupby('entry_hour').apply(
        lambda x: (x['profit_loss'] > 0).mean() * 100 if len(x) > 0 else 0
    )
    analysis['hourly'] = hourly_performance
    
    # Analysis by confidence level
    if 'confidence' in trades_df.columns:
        trades_df['confidence_bucket'] = pd.cut(
            trades_df['confidence'], 
            bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        )
        confidence_performance = trades_df.groupby('confidence_bucket')['profit_loss'].agg(['mean', 'sum', 'count'])
        confidence_performance['win_rate'] = trades_df.groupby('confidence_bucket').apply(
            lambda x: (x['profit_loss'] > 0).mean() * 100 if len(x) > 0 else 0
        )
        analysis['confidence'] = confidence_performance
    
    # Analysis by regime
    if 'regime' in trades_df.columns:
        regime_performance = trades_df.groupby('regime')['profit_loss'].agg(['mean', 'sum', 'count'])
        regime_performance['win_rate'] = trades_df.groupby('regime').apply(
            lambda x: (x['profit_loss'] > 0).mean() * 100 if len(x) > 0 else 0
        )
        analysis['regime'] = regime_performance
    
    # Analysis by exit reason
    if 'exit_reason' in trades_df.columns:
        reason_performance = trades_df.groupby('exit_reason')['profit_loss'].agg(['mean', 'sum', 'count'])
        reason_performance['win_rate'] = trades_df.groupby('exit_reason').apply(
            lambda x: (x['profit_loss'] > 0).mean() * 100 if len(x) > 0 else 0
        )
        analysis['exit_reason'] = reason_performance
    
    return analysis

# Function to simulate across multiple symbols
def multi_symbol_simulation(tickers, date_list, params=None):
    """
    Run simulations on multiple symbols and aggregate results
    """
    all_trades = []
    all_metrics = []
    
    for ticker in tickers:
        ticker_trades, ticker_metrics = multi_day_simulation(ticker, date_list, params)
        if not ticker_trades.empty:
            all_trades.append(ticker_trades)
        if not ticker_metrics.empty:
            all_metrics.append(ticker_metrics)
    
    # Combine results
    combined_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
    combined_metrics = pd.concat(all_metrics) if all_metrics else pd.DataFrame()
    
    return combined_trades, combined_metrics

# Function to integrate with paper trading system
def save_to_paper_trading(trades_df, date):
    """
    Push simulation trades to paper trading system
    """
    # Connect to DB for paper trading
    try:
        pt_conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
        pt_conn.autocommit = True
        cursor = pt_conn.cursor()
        
        # Get only trades from the specified date
        day_trades = trades_df[trades_df['date'] == date.date()]
        
        if not day_trades.empty:
            for _, trade in day_trades.iterrows():
                # Insert entry transaction
                entry_sql = """
                INSERT INTO paper_trades
                (entry_time, ticker, entry_price, shares, trade_status, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                entry_values = (
                    trade['entry_time'], 
                    trade['ticker'], 
                    float(trade['entry_price']), 
                    int(trade['shares']),
                    'SIMULATED',  # Special status
                    float(trade['confidence'])
                )
                
                cursor.execute(entry_sql, entry_values)
                
            print(f"Saved {len(day_trades)} simulation trades to paper trading system")
        
        cursor.close()
        pt_conn.close()
        
    except Exception as e:
        print(f"Error saving to paper trading: {e}")

# Main function to run the enhanced simulation
def run_enhanced_simulation():
    # 1. Extend Simulation Period
    test_dates = [
        pd.Timestamp('2025-05-02', tz='UTC'),
        pd.Timestamp('2025-05-09', tz='UTC'),
        pd.Timestamp('2025-05-16', tz='UTC')
    ]
    
    print("Running enhanced trading simulation with multiple improvements...")
    
    # Get user choice for mode
    print("\nChoose simulation mode:")
    print("1. Single-day simulation (visualize trades)")
    print("2. Multi-day simulation (one ticker)")
    print("3. Parameter optimization")
    print("4. Multi-symbol testing")
    print("5. Analyze trading patterns")
    print("6. Paper trading integration")
    print("7. Fetch missing data from Polygon")
    print("8. Model improvement through historical learning")
    print("9. Model calibration")
    
    try:
        choice = int(input("\nEnter your choice (1-9): "))
    except ValueError:
        choice = 1  # Default
    
    # For options that need data, ask if we should fetch missing data
    fetch_missing = False
    if choice in [1, 2, 4, 5]:
        fetch_missing = input("Automatically fetch missing data from Polygon? (y/n): ").lower() == 'y'
    
    if choice == 1:
        # Single day simulation with visualization
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
        date_str = input("Enter date (YYYY-MM-DD) (default: 2025-05-02): ") or "2025-05-02"
        date = pd.Timestamp(date_str, tz='UTC')
        
        # Add diagnostic mode option
        diagnostic_mode = input("Run in diagnostic mode? (y/n): ").lower() == 'y'
        
        # Get the best model path (will use calibrated if available)
        model_path = get_best_model_path(ticker)
        
        start_time = time.time()
        trades_df, metrics_df, viz_data = simulate_single_day(
            ticker, 
            date, 
            fetch_missing=fetch_missing, 
            diagnostic_mode=diagnostic_mode,
            custom_model_path=model_path
        )
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    
    elif choice == 2:
        # 1. Multi-day simulation
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
        
        print(f"\nRunning simulation for {ticker} across multiple days...")
        start_time = time.time()
        
        all_trades, all_metrics = multi_day_simulation(ticker, test_dates)
        
        print(f"Multi-day simulation completed in {time.time() - start_time:.2f} seconds")
        
        if not all_metrics.empty:
            # Show summary statistics
            print("\n--- Multi-Day Trading Summary ---")
            print(f"Total Days: {len(all_metrics)}")
            print(f"Total Trades: {len(all_trades)}")
            
            # Calculate aggregated metrics
            total_pnl = all_metrics['total_pnl'].sum()
            avg_daily_return = all_metrics['return_pct'].mean()
            win_days = (all_metrics['total_pnl'] > 0).sum()
            win_rate_days = win_days / len(all_metrics) * 100 if len(all_metrics) > 0 else 0
            
            print(f"Total P&L: ${total_pnl:.2f}")
            print(f"Average Daily Return: {avg_daily_return:.2f}%")
            print(f"Winning Days: {win_days}/{len(all_metrics)} ({win_rate_days:.2f}%)")
            
            if not all_trades.empty:
                winning_trades = all_trades[all_trades['profit_loss'] > 0]
                win_rate_trades = len(winning_trades) / len(all_trades) * 100
                
                print(f"Overall Win Rate (Trades): {win_rate_trades:.2f}%")
                print(f"Average Profit per Trade: ${all_trades['profit_loss'].mean():.2f}")
            
            # Create equity curve over multiple days
            initial_equity = DEFAULT_PARAMS['capital']
            daily_returns = all_metrics['return_pct'].values / 100
            equity_curve = [initial_equity]
            
            for ret in daily_returns:
                equity_curve.append(equity_curve[-1] * (1 + ret))
            
            plt.figure(figsize=(10, 6))
            plt.plot(all_metrics['date'], equity_curve[1:], 'b-', marker='o')
            plt.title(f'{ticker} Multi-Day Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_filename = f"{ticker}_multi_day_equity_curve.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"\nEquity curve saved to {plot_filename}")
            
            # Save results to CSV
            metrics_csv = f"{ticker}_multi_day_metrics.csv"
            trades_csv = f"{ticker}_multi_day_trades.csv"
            
            all_metrics.to_csv(metrics_csv, index=False)
            all_trades.to_csv(trades_csv, index=False)
            
            print(f"Results saved to {metrics_csv} and {trades_csv}")
            
            # 5. Analyze Trading Patterns (across all days)
            print("\n--- Trading Pattern Analysis (All Days) ---")
            analysis = analyze_trading_patterns(all_trades)
            
            if 'hourly' in analysis:
                print("\nPerformance by Hour:")
                print(analysis['hourly'])
            
            if 'confidence' in analysis:
                print("\nPerformance by Confidence Level:")
                print(analysis['confidence'])
                
            if 'regime' in analysis:
                print("\nPerformance by Market Regime:")
                print(analysis['regime'])
            
            # Use the utility function in the Excel export code
            excel_trades = prepare_dataframe_for_excel(all_trades)

            with pd.ExcelWriter(f"{ticker}_trading_pattern_analysis.xlsx") as writer:
                for key, df in analysis.items():
                    df.to_excel(writer, sheet_name=key)
                excel_trades.to_excel(writer, sheet_name='all_trades')
        else:
            print("No trading results generated for any day")
    
    elif choice == 3:
        # 2. Parameter Optimization
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
        date_str = input("Enter date (YYYY-MM-DD) (default: 2025-05-02): ") or "2025-05-02"
        date = pd.Timestamp(date_str, tz='UTC')
        
        # Define parameter grid with more granular values
        param_grid = {
            'confidence_threshold': [0.53, 0.55, 0.57, 0.59, 0.60],
            'target_profit_pct': [0.01, 0.015, 0.02, 0.025, 0.03],
            'stop_loss_pct': [0.005, 0.0075, 0.01, 0.0125, 0.015]
        }
        
        # Add time-of-day based trading restriction as an option
        include_time_filter = input("Include time-of-day filter? (y/n): ").lower() == 'y'
        if include_time_filter:
            param_grid['trade_start_hour'] = [9, 10, 11]
            param_grid['trade_end_hour'] = [14, 15, 16]
        
        print(f"\nRunning parameter optimization for {ticker} on {date.date()}...")
        print(f"Testing {len(list(itertools.product(*param_grid.values())))} parameter combinations")
        
        start_time = time.time()
        optimization_results = optimize_parameters(ticker, date, param_grid)
        print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        
        if not optimization_results.empty:
            try:
                # Find the best parameter combinations
                best_return_idx = optimization_results['return_pct'].idxmax()
                if isinstance(best_return_idx, pd.Series):
                    best_return_idx = best_return_idx.iloc[0]  # Take first if multiple maxima
                
                best_sharpe_idx = optimization_results['sharpe_ratio'].idxmax()
                if isinstance(best_sharpe_idx, pd.Series):
                    best_sharpe_idx = best_sharpe_idx.iloc[0]
                
                best_win_rate_idx = optimization_results['win_rate'].idxmax()
                if isinstance(best_win_rate_idx, pd.Series):
                    best_win_rate_idx = best_win_rate_idx.iloc[0]
                
                # Access rows directly
                best_return_row = optimization_results.loc[best_return_idx]
                best_sharpe_row = optimization_results.loc[best_sharpe_idx]
                best_win_rate_row = optimization_results.loc[best_win_rate_idx]
                
                print("\n--- Optimization Results ---")
                
                print("\nBest Parameters by Return:")
                for param in param_grid.keys():
                    param_value = best_return_row[param]
                    if isinstance(param_value, pd.Series):
                        param_value = param_value.iloc[0]  # Get first value if Series
                    print(f"{param}: {param_value}")
                
                return_value = best_return_row['return_pct']
                if isinstance(return_value, pd.Series):
                    return_value = return_value.iloc[0]
                print(f"Return: {float(return_value):.2f}%")
                
                win_rate_value = best_return_row['win_rate']
                if isinstance(win_rate_value, pd.Series):
                    win_rate_value = win_rate_value.iloc[0]
                print(f"Win Rate: {float(win_rate_value) if not pd.isna(win_rate_value) else 0:.2f}%")
                
                trade_count_value = best_return_row['trade_count']
                if isinstance(trade_count_value, pd.Series):
                    trade_count_value = trade_count_value.iloc[0]
                print(f"Trades: {int(trade_count_value)}")
                
                print("\nBest Parameters by Sharpe Ratio:")
                for param in param_grid.keys():
                    param_value = best_sharpe_row[param]
                    if isinstance(param_value, pd.Series):
                        param_value = param_value.iloc[0]
                    print(f"{param}: {param_value}")
                
                sharpe_value = best_sharpe_row['sharpe_ratio']
                if isinstance(sharpe_value, pd.Series):
                    sharpe_value = sharpe_value.iloc[0]
                print(f"Sharpe Ratio: {float(sharpe_value) if not pd.isna(sharpe_value) else 0:.2f}")
                
                return_value = best_sharpe_row['return_pct']
                if isinstance(return_value, pd.Series):
                    return_value = return_value.iloc[0]
                print(f"Return: {float(return_value):.2f}%")
                
                win_rate_value = best_sharpe_row['win_rate']
                if isinstance(win_rate_value, pd.Series):
                    win_rate_value = win_rate_value.iloc[0]
                print(f"Win Rate: {float(win_rate_value) if not pd.isna(win_rate_value) else 0:.2f}%")
                
                trade_count_value = best_sharpe_row['trade_count']
                if isinstance(trade_count_value, pd.Series):
                    trade_count_value = trade_count_value.iloc[0]
                print(f"Trades: {int(trade_count_value)}")
                
                print("\nBest Parameters by Win Rate:")
                for param in param_grid.keys():
                    param_value = best_win_rate_row[param]
                    if isinstance(param_value, pd.Series):
                        param_value = param_value.iloc[0]
                    print(f"{param}: {param_value}")
                
                win_rate_value = best_win_rate_row['win_rate']
                if isinstance(win_rate_value, pd.Series):
                    win_rate_value = win_rate_value.iloc[0]
                print(f"Win Rate: {float(win_rate_value) if not pd.isna(win_rate_value) else 0:.2f}%")
                
                return_value = best_win_rate_row['return_pct']
                if isinstance(return_value, pd.Series):
                    return_value = return_value.iloc[0]
                print(f"Return: {float(return_value):.2f}%")
                
                trade_count_value = best_win_rate_row['trade_count']
                if isinstance(trade_count_value, pd.Series):
                    trade_count_value = trade_count_value.iloc[0]
                print(f"Trades: {int(trade_count_value)}")
                
                # Save optimization results
                csv_filename = f"{ticker}_parameter_optimization_{date.date()}.csv"
                optimization_results.to_csv(csv_filename, index=False)
                print(f"\nOptimization results saved to {csv_filename}")
                
                # Create visualization of parameter optimization results
                try:
                    plt.figure(figsize=(15, 10))
                    
                    # 1. Confidence threshold vs return
                    plt.subplot(2, 3, 1)
                    pivot_confidence = optimization_results.pivot_table(
                        values='return_pct', 
                        index='target_profit_pct', 
                        columns='confidence_threshold',
                        aggfunc='mean'
                    )
                    sns.heatmap(pivot_confidence, annot=True, cmap='viridis', fmt='.2f')
                    plt.title('Returns by Confidence & Target Profit')
                    
                    # 2. Profit vs Stop Loss
                    plt.subplot(2, 3, 2)
                    pivot_stops = optimization_results.pivot_table(
                        values='return_pct', 
                        index='target_profit_pct', 
                        columns='stop_loss_pct',
                        aggfunc='mean'
                    )
                    sns.heatmap(pivot_stops, annot=True, cmap='viridis', fmt='.2f')
                    plt.title('Returns by Target & Stop Loss')
                    
                    # 3. Win Rate Analysis
                    plt.subplot(2, 3, 3)
                    pivot_winrate = optimization_results.pivot_table(
                        values='win_rate', 
                        index='target_profit_pct', 
                        columns='stop_loss_pct',
                        aggfunc='mean'
                    )
                    sns.heatmap(pivot_winrate, annot=True, cmap='RdYlGn', fmt='.1f')
                    plt.title('Win Rate by Target & Stop Loss')
                    
                    # 4. Trade count analysis
                    plt.subplot(2, 3, 4)
                    pivot_trades = optimization_results.pivot_table(
                        values='trade_count', 
                        index='confidence_threshold', 
                        columns='target_profit_pct',
                        aggfunc='mean'
                    )
                    sns.heatmap(pivot_trades, annot=True, cmap='Blues', fmt='.0f')
                    plt.title('Trade Count by Confidence & Target')
                    
                    # 5. Top 5 parameter combinations
                    plt.subplot(2, 3, 5)
                    top_combos = optimization_results.sort_values('return_pct', ascending=False).head(5)
                    param_names = param_grid.keys()
                    combo_labels = [
                        f"C:{row['confidence_threshold']:.2f}, T:{row['target_profit_pct']:.3f}, S:{row['stop_loss_pct']:.3f}"
                        for _, row in top_combos.iterrows()
                    ]
                    plt.barh(combo_labels, top_combos['return_pct'])
                    plt.xlabel('Return %')
                    plt.title('Top 5 Parameter Combinations')
                    plt.grid(axis='x', alpha=0.3)
                    
                    # Save the figure
                    plt.tight_layout()
                    viz_filename = f"{ticker}_parameter_optimization_viz_{date.date()}.png"
                    plt.savefig(viz_filename)
                    plt.close()
                    print(f"Parameter visualization saved to {viz_filename}")
                except Exception as e:
                    print(f"Error creating parameter visualization: {e}")
                
                # Run simulation with best parameters and create dashboard
                print("\nRunning simulation with optimal parameters...")
                best_params = {}
                for param in param_grid.keys():
                    best_params[param] = best_return_row[param]
                    if isinstance(best_params[param], pd.Series):
                        best_params[param] = best_params[param].iloc[0]
                
                best_trades_df, best_metrics_df, best_viz_data = simulate_single_day(ticker, date, best_params)
                
                if best_trades_df is not None and not best_trades_df.empty:
                    # Create dashboard with multiple plots
                    plt.figure(figsize=(15, 12))
                    
                    # 1. Price chart with trades
                    ax1 = plt.subplot(3, 1, 1)
                    ax1.plot(best_viz_data['price_data'].index, best_viz_data['price_data'], 'k-', label=f'{ticker} Price')
                    
                    # Plot entry and exit points 
                    for i in range(len(best_viz_data['trade_entries'])):
                        entry_time, entry_price = best_viz_data['trade_entries'][i]
                        exit_time, exit_price = best_viz_data['trade_exits'][i]
                        trade_type = best_viz_data['trade_types'][i]
                        
                        # Color code by trade outcome
                        if trade_type == 'win':
                            entry_color = 'blue'
                            exit_color = 'green'
                            linestyle = '--'
                        elif trade_type == 'loss':
                            entry_color = 'blue'
                            exit_color = 'red'
                            linestyle = '--'
                        else:  # hold
                            entry_color = 'blue'
                            exit_color = 'black'
                            linestyle = ':'
                            
                        # Plot entry and exit
                        ax1.scatter(entry_time, entry_price, color=entry_color, marker='^', s=100)
                        ax1.scatter(exit_time, exit_price, color=exit_color, marker='v', s=100)
                        ax1.plot([entry_time, exit_time], [entry_price, exit_price], linestyle=linestyle, color=entry_color, alpha=0.5)
                    
                    ax1.set_title(f"{ticker} Trading with Optimal Parameters - P&L: ${best_metrics_df['total_pnl'].iloc[0]:.2f}")
                    ax1.set_ylabel('Price ($)')
                    ax1.grid(True, alpha=0.3)
                    
                    # 2. Equity curve
                    ax2 = plt.subplot(3, 1, 2)
                    ax2.plot(best_viz_data['equity_curve'].index, best_viz_data['equity_curve'], 'b-', label='Equity Curve')
                    ax2.set_title('Equity Curve')
                    ax2.set_ylabel('Equity ($)')
                    ax2.grid(True, alpha=0.3)
                    
                    # 3. Trade analysis
                    ax3 = plt.subplot(3, 2, 5)
                    trade_results = ['Win' if p > 0 else 'Loss' for p in best_trades_df['profit_loss']]
                    result_counts = pd.Series(trade_results).value_counts()
                    ax3.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', colors=['green', 'red'])
                    ax3.set_title('Win/Loss Distribution')
                    
                    # 4. Exit reasons
                    ax4 = plt.subplot(3, 2, 6)
                    exit_counts = best_trades_df['exit_reason'].value_counts()
                    ax4.bar(exit_counts.index, exit_counts.values)
                    ax4.set_title('Exit Reasons')
                    ax4.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    dashboard_filename = f"{ticker}_optimal_params_dashboard_{date.date()}.png"
                    plt.savefig(dashboard_filename)
                    plt.close()
                    print(f"Optimal parameters dashboard saved to {dashboard_filename}")
            except Exception as e:
                print(f"Error processing optimization results: {e}")
                print("Full optimization results have been saved to CSV for manual inspection.")
                csv_filename = f"{ticker}_parameter_optimization_{date.date()}_error.csv"
                optimization_results.to_csv(csv_filename, index=False)
                print(f"Results saved to {csv_filename}")
        else:
            print("Optimization did not produce any valid results")
            
    elif choice == 4:
        # 8. Multi-Symbol Testing
        tickers_input = input("Enter tickers separated by commas (default: TSLA,AAPL,MSFT,AMZN,NVDA): ")
        if tickers_input:
            tickers = [t.strip() for t in tickers_input.split(',')]
        else:
            tickers = TICKERS_TO_TEST
        
        print(f"\nRunning simulation for {len(tickers)} tickers: {', '.join(tickers)}")
        date_str = input("Enter date (YYYY-MM-DD) (default: 2025-05-02): ") or "2025-05-02"
        test_date = pd.Timestamp(date_str, tz='UTC')
        
        # Single day, multiple symbols
        start_time = time.time()
        
        all_trades = []
        all_metrics = []
        
        for ticker in tickers:
            print(f"\nSimulating {ticker} on {test_date.date()}...")
            trades_df, metrics_df, _ = simulate_single_day(ticker, test_date)
            
            # Add null checks before checking if empty
            if trades_df is not None and not trades_df.empty:
                all_trades.append(trades_df)
            if metrics_df is not None and not metrics_df.empty:
                all_metrics.append(metrics_df)
        
        print(f"Multi-symbol simulation completed in {time.time() - start_time:.2f} seconds")
        
        # Check if we have any metrics at all
        if all_metrics:
            combined_metrics = pd.concat(all_metrics)
            
            print("\n--- Multi-Symbol Performance Comparison ---")
            sorted_metrics = combined_metrics.sort_values('return_pct', ascending=False)
            
            # Display key metrics
            for _, row in sorted_metrics.iterrows():
                ticker = row['ticker']
                return_pct = row['return_pct']
                trade_count = row['trade_count']
                win_rate = row['win_rate'] if not pd.isna(row['win_rate']) else 0
                
                print(f"{ticker}: Return {return_pct:.2f}%, {trade_count} trades, Win Rate {win_rate:.2f}%")
            
            # Create comparison chart
            plt.figure(figsize=(12, 6))
            bars = plt.bar(sorted_metrics['ticker'], sorted_metrics['return_pct'])
            
            # Color bars based on positive/negative returns
            for i, bar in enumerate(bars):
                if sorted_metrics['return_pct'].iloc[i] >= 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            plt.title(f'Multi-Symbol Performance Comparison ({test_date.date()})')
            plt.xlabel('Ticker')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plot_filename = f"multi_symbol_comparison_{test_date.date()}.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"\nComparison chart saved to {plot_filename}")
            
            # Save results
            if all_trades:
                combined_trades = pd.concat(all_trades)
                trades_csv = f"multi_symbol_trades_{test_date.date()}.csv"
                combined_trades.to_csv(trades_csv, index=False)
                print(f"Detailed trades saved to {trades_csv}")
            
            metrics_csv = f"multi_symbol_metrics_{test_date.date()}.csv"
            combined_metrics.to_csv(metrics_csv, index=False)
            print(f"Metrics saved to {metrics_csv}")
        else:
            print("No tradable results from any symbol")
    
    elif choice == 5:
        # Analyze Trading Patterns (specific date range for deeper analysis)
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
        start_date_str = input("Enter start date (YYYY-MM-DD) (default: 2025-05-01): ") or "2025-05-01"
        end_date_str = input("Enter end date (YYYY-MM-DD) (default: 2025-05-20): ") or "2025-05-20"
        
        start_date = pd.Timestamp(start_date_str, tz='UTC')
        end_date = pd.Timestamp(end_date_str, tz='UTC')
        
        # Generate list of business days
        business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        business_days = [day for day in business_days if day.dayofweek < 5]  # Exclude weekends
        
        print(f"\nAnalyzing {ticker} trading patterns from {start_date_str} to {end_date_str} ({len(business_days)} trading days)...")
        
        start_time = time.time()
        all_trades, _ = multi_day_simulation(ticker, business_days)
        print(f"Analysis data collection completed in {time.time() - start_time:.2f} seconds")
        
        if not all_trades.empty:
            print(f"\nAnalyzed {len(all_trades)} trades across {len(business_days)} trading days")
            
            # Generate comprehensive pattern analysis
            analysis = analyze_trading_patterns(all_trades)
            
            # More detailed time analysis
            all_trades['entry_time_minute'] = all_trades['entry_time'].dt.minute
            all_trades['entry_hour_minute'] = all_trades['entry_time'].dt.hour + all_trades['entry_time'].dt.minute / 60
            
            # Plot patterns
            plt.figure(figsize=(15, 10))
            
            # 1. Trades by hour
            ax1 = plt.subplot(2, 2, 1)
            hourly_trades = all_trades.groupby('entry_hour').size()
            hourly_win_rate = all_trades.groupby('entry_hour').apply(
                lambda x: (x['profit_loss'] > 0).mean() * 100 if len(x) > 0 else 0
            )
            
            ax1.bar(hourly_trades.index, hourly_trades, alpha=0.7, label='Trade Count')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(hourly_win_rate.index, hourly_win_rate, 'r-o', label='Win Rate (%)')
            ax1.set_title('Trades by Hour')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Number of Trades')
            ax1_twin.set_ylabel('Win Rate (%)')
            ax1.set_xticks(range(9, 17))
            ax1.grid(True, alpha=0.3)
            
            # 2. Trades by confidence level
            if 'confidence' in all_trades.columns:
                ax2 = plt.subplot(2, 2, 2)
                confidence_trades = all_trades.groupby('confidence_bucket').size()
                confidence_win_rate = all_trades.groupby('confidence_bucket').apply(
                    lambda x: (x['profit_loss'] > 0).mean() * 100 if len(x) > 0 else 0
                )
                
                ax2.bar(confidence_trades.index, confidence_trades, alpha=0.7, label='Trade Count')
                ax2_twin = ax2.twinx()
                ax2_twin.plot(range(len(confidence_win_rate)), confidence_win_rate.values, 'r-o', label='Win Rate (%)')
                ax2.set_title('Trades by Confidence Level')
                ax2.set_xlabel('Confidence Bucket')
                ax2.set_ylabel('Number of Trades')
                ax2_twin.set_ylabel('Win Rate (%)')
                ax2.set_xticks(range(len(confidence_trades.index)))
                ax2.set_xticklabels(confidence_trades.index, rotation=45)
                ax2.grid(True, alpha=0.3)
            
            # 3. Trades by regime
            if 'regime' in all_trades.columns:
                ax3 = plt.subplot(2, 2, 3)
                regime_trades = all_trades.groupby('regime').size()
                regime_win_rate = all_trades.groupby('regime').apply(
                    lambda x: (x['profit_loss'] > 0).mean() * 100 if len(x) > 0 else 0
                )
                
                ax3.bar(regime_trades.index, regime_trades, alpha=0.7, label='Trade Count')
                ax3_twin = ax3.twinx()
                ax3_twin.plot(range(len(regime_win_rate)), regime_win_rate.values, 'r-o', label='Win Rate (%)')
                ax3.set_title('Trades by Market Regime')
                ax3.set_xlabel('Regime')
                ax3.set_ylabel('Number of Trades')
                ax3_twin.set_ylabel('Win Rate (%)')
                ax3.set_xticks(range(len(regime_trades.index)))
                ax3.set_xticklabels(regime_trades.index, rotation=45)
                ax3.grid(True, alpha=0.3)
            
            # 4. Trades by exit reason
            if 'exit_reason' in all_trades.columns:
                ax4 = plt.subplot(2, 2, 4)
                reason_trades = all_trades.groupby('exit_reason').size()
                reason_win_rate = all_trades.groupby('exit_reason').apply(
                    lambda x: (x['profit_loss'] > 0).mean() * 100 if len(x) > 0 else 0
                )
                
                ax4.bar(reason_trades.index, reason_trades, alpha=0.7, label='Trade Count')
                ax4_twin = ax4.twinx()
                ax4_twin.plot(range(len(reason_win_rate)), reason_win_rate.values, 'r-o', label='Win Rate (%)')
                ax4.set_title('Trades by Exit Reason')
                ax4.set_xlabel('Exit Reason')
                ax4.set_ylabel('Number of Trades')
                ax4_twin.set_ylabel('Win Rate (%)')
                ax4.set_xticks(range(len(reason_trades.index)))
                ax4.set_xticklabels(reason_trades.index, rotation=45)
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = f"{ticker}_trading_pattern_analysis.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"\nTrading pattern analysis visualization saved to {plot_filename}")
            
            # Save detailed analysis to CSV
            csv_filename = f"{ticker}_trading_pattern_analysis.csv"
            
            # Combine analysis results into a single file with sheets
            with pd.ExcelWriter(f"{ticker}_trading_pattern_analysis.xlsx") as writer:
                for key, df in analysis.items():
                    df.to_excel(writer, sheet_name=key)
                excel_trades = prepare_dataframe_for_excel(all_trades)
                excel_trades.to_excel(writer, sheet_name='all_trades')
            
            print(f"Detailed trading pattern analysis saved to {ticker}_trading_pattern_analysis.xlsx")
            
            # Print summary of best periods/conditions
            if 'hourly' in analysis:
                best_hour = analysis['hourly']['win_rate'].idxmax()
                print(f"\nBest hour for trading: {best_hour}:00 (Win Rate: {analysis['hourly']['win_rate'].max():.2f}%)")
            
            if 'confidence' in analysis:
                best_confidence = analysis['confidence']['win_rate'].idxmax()
                print(f"Best confidence range: {best_confidence} (Win Rate: {analysis['confidence']['win_rate'].max():.2f}%)")
                
            if 'regime' in analysis:
                best_regime = analysis['regime']['win_rate'].idxmax()
                print(f"Best market regime: {best_regime} (Win Rate: {analysis['regime']['win_rate'].max():.2f}%)")
        else:
            print("No trades found for analysis")
    
    elif choice == 6:
        # Paper Trading Integration
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
        date_str = input("Enter date (YYYY-MM-DD) (default: 2025-05-02): ") or "2025-05-02"
        date = pd.Timestamp(date_str, tz='UTC')
        
        print(f"\nRunning simulation and integrating with paper trading for {ticker} on {date.date()}...")
        
        # Get the best model path (will use calibrated if available)
        model_path = get_best_model_path(ticker)
        
        start_time = time.time()
        trades_df, metrics_df, _ = simulate_single_day(
            ticker, 
            date,
            custom_model_path=model_path
        )
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
        
        if not trades_df.empty:
            # Display results
            print(f"\nSimulated {len(trades_df)} trades with result: ${metrics_df['total_pnl'].iloc[0]:.2f} ({metrics_df['return_pct'].iloc[0]:.2f}%)")
            
            # Ask if user wants to save to paper trading system
            save_to_pt = input("\nSave these simulation results to paper trading system? (y/n): ").lower()
            
            if save_to_pt == 'y':
                save_to_paper_trading(trades_df, date)
        else:
            print("No trades executed during simulation")
    
    elif choice == 7:
        # Directly fetch data from Polygon
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
        date_str = input("Enter date (YYYY-MM-DD) (default: 2025-05-02): ") or "2025-05-02"
        date = pd.Timestamp(date_str, tz='UTC')
        
        print(f"Fetching data for {ticker} on {date.date()} from Polygon API...")
        success = fetch_missing_data(ticker, date)
        
        if success:
            print(f"Successfully fetched and saved data for {ticker} on {date.date()}")
            
            # Ask if user wants to run a simulation with the newly fetched data
            run_sim = input("Run simulation with the newly fetched data? (y/n): ").lower() == 'y'
            if run_sim:
                trades_df, metrics_df, viz_data = simulate_single_day(ticker, date)
                
                if trades_df is not None and not trades_df.empty:
                    print(f"\nSimulated {len(trades_df)} trades with result: ${metrics_df['total_pnl'].iloc[0]:.2f}")
                else:
                    print("No trades were executed in the simulation")
        else:
            print(f"Failed to fetch data for {ticker} on {date.date()}")
    
    elif choice == 8:
        # Model improvement through historical learning
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
        
        # Validate ticker - ensure it's not just a number and follows stock symbol pattern
        if ticker.isdigit() or len(ticker) < 1 or len(ticker) > 5:
            print(f"Warning: '{ticker}' doesn't appear to be a valid stock symbol.")
            confirm = input("Do you want to continue anyway? (y/n): ").lower()
            if confirm != 'y':
                print("Operation cancelled. Please try again with a valid ticker symbol.")
                return
        
        months = int(input("How many months of data to use (default: 3): ") or "3")
        iterations = int(input("Number of learning iterations (default: 3): ") or "3")
        
        print(f"\nStarting model improvement process for {ticker} using {months} months of data...")
        
        # Make sure required modules are imported
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            import numpy as np
            import lightgbm as lgb
        except ImportError as e:
            print(f"Missing required package: {e}")
            print("Please install required packages: pip install scikit-learn lightgbm")
            return
        
        start_time = time.time()
        
        # Start the learning process - with improved error handling
        try:
            result = learn_from_past_trading_mistakes(ticker, months, iterations)
            if result is None:
                print(f"Could not improve model for {ticker} due to insufficient historical data.")
                return
            else:
                model, scaler, features = result
                
                print(f"\nModel improvement process completed in {time.time() - start_time:.2f} seconds")
                
                # Make the improved model the default model
                if model is not None:
                    final_model_path = f'lgbm_final_model_enhanced_{ticker.lower()}.joblib'
                    joblib.dump((model, scaler, features), final_model_path)
                    print(f"Final improved model saved as the default model: {final_model_path}")
        except Exception as e:
            print(f"Error during model improvement process: {e}")
            print("Please check if the ticker symbol is valid and has sufficient historical data.")
    
# Replace the existing choice == 9 block with:

    elif choice == 9:
        # Model Calibration
        print("\n--- Model Calibration Tool ---")
        print("This will calibrate your model to produce better confidence scores.")
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
    
        try:
            # First check if model_calibration.py exists
            if not os.path.exists("model_calibration.py"):
                print("Error: model_calibration.py file not found.")
                print("Creating the file now...")
                with open("model_calibration.py", "w") as f:
                    # Fetch template from the previous code
                    # This is a simplified version for this demo
                    f.write("""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta, timezone
import os
from lgbm_prediction_service import create_features
import psycopg2
from dotenv import load_dotenv

# Load environment variables for database connection
load_dotenv()
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "polygondata"
DB_USER = "polygonuser"

def suggest_optimal_threshold(calibrated_probs):
    \"\"\"Suggest an optimal threshold based on the distribution of calibrated probabilities\"\"\"
    # Get distribution statistics
    mean_prob = np.mean(calibrated_probs)
    max_prob = np.max(calibrated_probs)
    
    # Find the 90th percentile as a reasonable threshold
    p90 = np.percentile(calibrated_probs, 90)
    
    # Suggest a threshold slightly below the 90th percentile
    # but not less than half the maximum probability
    suggested = min(p90, max(max_prob * 0.5, mean_prob * 2))
    
    print(f"Suggested threshold for calibrated model: {suggested:.4f}")
    print(f"This would activate on approximately 10% of predictions")
    
    return suggested

# Other functions from the complete model_calibration.py file...
                """)
                print("Created a simple model_calibration.py template. You will need to complete it with all functions.")
            
            # Import the model_calibration module
            import model_calibration
        
            method = input("Calibration method (isotonic/platt) default: isotonic: ") or "isotonic"
            result = model_calibration.calibrate_model(ticker, method)
        
            if result:
                print("\nModel calibration successful!")
                use_now = input("Use calibrated model for a simulation now? (y/n): ").lower() == 'y'
                if use_now:
                    date_str = input("Enter date (YYYY-MM-DD) (default: 2025-05-02): ") or "2025-05-02"
                    date = pd.Timestamp(date_str, tz='UTC')
                
                    # Specify the calibrated model path
                    calibrated_model_path = f'lgbm_calibrated_model_{ticker.lower()}.joblib'
                
                    # First run diagnostic mode to get probability distribution
                    print("Running diagnostic to determine optimal threshold...")
                    cal_params = DEFAULT_PARAMS.copy()
                
                    # Load the calibrated model to analyze its probabilities
                    calibration_package = joblib.load(calibrated_model_path)
                
                    # Run an initial simulation to get the probability distribution
                    df_raw = fetch_data(ticker, date)
                    if not df_raw.empty:
                        # Generate features
                        df_features = create_features(df_raw.copy())
                    
                        # Create a copy for model features
                        df_features_model = df_features.copy()
                        df_features_model = df_features_model.dropna()
                    
                        # Get features for prediction
                        features_to_use = calibration_package['features']
                        features_for_prediction = df_features_model[features_to_use].copy()
                    
                        # Scale features
                        scaler = calibration_package['scaler']
                        df_features_scaled = scaler.transform(features_for_prediction)
                    
                        # Get predictions
                        from model_calibration import predict_with_calibrated_model
                        predictions = predict_with_calibrated_model(calibration_package, df_features_scaled)
                    
                        # Determine optimal threshold
                        if hasattr(model_calibration, 'suggest_optimal_threshold'):
                            optimal_threshold = model_calibration.suggest_optimal_threshold(predictions)
                        else:
                            # Fallback to a simple percentile if the function doesn't exist
                            optimal_threshold = np.percentile(predictions, 90)
                            print(f"Using 90th percentile as threshold: {optimal_threshold:.4f}")
                    
                        # Update parameters with optimal threshold
                        cal_params['confidence_threshold'] = optimal_threshold
                    else:
                        # Fallback if we can't get predictions
                        cal_params['confidence_threshold'] = 0.05
                        print(f"No data available for {ticker} on {date}. Using fallback threshold: {cal_params['confidence_threshold']}")
                
                    # Now run the actual simulation with the optimal threshold
                    start_time = time.time()
                    print(f"Running simulation with calibrated model and threshold {cal_params['confidence_threshold']:.4f}...")
                    trades_df, metrics_df, viz_data = simulate_single_day(
                        ticker, 
                        date, 
                        params=cal_params,
                        diagnostic_mode=False,
                        custom_model_path=calibrated_model_path
                    )
                    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
            else:
                print("Model calibration failed.")
        except Exception as e:
            print(f"Error during model calibration: {e}")
            import traceback
            traceback.print_exc() 
    else:
        print("Invalid choice. Please run again and select options 1-9.")
    
    # Clean up
    if conn:
        conn.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    run_enhanced_simulation()

