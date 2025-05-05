import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta, timezone

# Import local modules
from lgbm_prediction_service import create_features
import os
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
    """Suggest an optimal threshold based on the distribution of calibrated probabilities"""
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

def fetch_calibration_data(ticker, months=1):
    """Fetch data for model calibration"""
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    conn.autocommit = True
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30*months)
    
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
        
        if df.duplicated(subset=['time']).any():
            print(f"Warning: Duplicate timestamps found in data for {ticker}.")
            df = df.drop_duplicates(subset=['time'], keep='first')
            
        df.set_index('time', inplace=True)
        print(f"Successfully fetched {len(df)} data points.")
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        if conn:
            conn.close()
        return None

def create_calibration_targets(df, lookahead_period=15, target_gain=1.0):
    """Create binary targets for calibration"""
    df_with_target = df.copy()
    
    # Calculate future returns for the lookahead period
    future_returns = df['close'].pct_change(periods=lookahead_period).shift(-lookahead_period) * 100
    
    # Create binary targets
    binary_target = (future_returns >= target_gain).astype(int)
    
    df_with_target['target'] = binary_target
    
    # Remove rows where we don't have targets
    df_with_target = df_with_target.dropna(subset=['target'])
    
    # Show class distribution
    target_distribution = df_with_target['target'].value_counts(normalize=True) * 100
    print(f"Target distribution: {target_gain}% gain opportunities: {target_distribution[1]:.2f}%")
    
    return df_with_target

def calibrate_model(ticker, method='isotonic'):
    """Calibrate a model to improve confidence scores"""
    print(f"Loading model for {ticker}...")
    model_path = f'lgbm_final_model_enhanced_{ticker.lower()}.joblib'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return False
    
    try:
        # Load the original model
        model, scaler, features = joblib.load(model_path)
        print("Original model loaded. Now generating calibration data...")
        
        # Fetch data for calibration
        calibration_data = fetch_calibration_data(ticker)
        if calibration_data is None or calibration_data.empty:
            print("Could not fetch calibration data.")
            return False
        
        # Generate features
        print("Generating features...")
        feature_df = create_features(calibration_data)
        
        # Create target labels
        labeled_df = create_calibration_targets(feature_df)
        
        # Prepare calibration data
        X = labeled_df.drop(['target', 'open', 'high', 'low', 'close', 'volume', 'vwap'], axis=1)
        y = labeled_df['target']
        
        # Handle NaNs
        X = X.fillna(0)
        
        # Ensure we only use the features the model was trained on
        X = X[features]
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Get original probabilities
        orig_probs = model.predict_proba(X_scaled)[:, 1]
        
        # Create calibration model
        print("Creating calibration model...")
        if method.lower() == 'isotonic':
            cal_model = IsotonicRegression(out_of_bounds='clip')
            cal_model.fit(orig_probs, y)
        else:  # platt scaling (logistic regression)
            cal_model = LogisticRegression(C=1.0)
            # Reshape probs to 2D array
            cal_model.fit(orig_probs.reshape(-1, 1), y)
        
        # Get calibrated probabilities
        if method.lower() == 'isotonic':
            cal_probs = cal_model.predict(orig_probs)
        else:
            cal_probs = cal_model.predict_proba(orig_probs.reshape(-1, 1))[:, 1]
        
        # Visualize calibration
        plt.figure(figsize=(10, 6))
        
        # Plot probabilities comparison
        plt.subplot(1, 2, 1)
        plt.hist(orig_probs, bins=20, alpha=0.5, label='Original')
        plt.hist(cal_probs, bins=20, alpha=0.5, label='Calibrated')
        plt.title('Probability Distributions')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot calibration curve
        plt.subplot(1, 2, 2)
        prob_true, prob_pred = calibration_curve(y, orig_probs, n_bins=10)
        cal_true, cal_pred = calibration_curve(y, cal_probs, n_bins=10)
        
        plt.plot(prob_pred, prob_true, 's-', label='Original')
        plt.plot(cal_pred, cal_true, 's-', label='Calibrated')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{ticker}_probability_calibration.png")
        plt.close()
        print(f"Original vs calibrated probabilities comparison saved to {ticker}_probability_calibration.png")
        
        # Print statistics
        print(f"Original probabilities - Mean: {orig_probs.mean():.4f}, Max: {orig_probs.max():.4f}")
        print(f"Calibrated probabilities - Mean: {cal_probs.mean():.4f}, Max: {cal_probs.max():.4f}")
        
        # Save the calibrated model
        calibrated_model_path = f'lgbm_calibrated_model_{ticker.lower()}.joblib'
        
        # Create a dictionary with all components needed
        optimal_threshold = suggest_optimal_threshold(cal_probs)
        calibration_package = {
            'original_model': model,
            'scaler': scaler,
            'features': features,
            'calibration_model': cal_model,
            'calibration_method': method,
            'calibration_date': datetime.now().strftime('%Y-%m-%d'),
            'optimal_threshold': optimal_threshold
        }
        
        # Save the package
        joblib.dump(calibration_package, calibrated_model_path)
        print(f"Calibrated model saved to {calibrated_model_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during model calibration: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_with_calibrated_model(calibration_package, X_scaled):
    """Make predictions using a calibrated model"""
    # Extract components
    original_model = calibration_package['original_model']
    calibration_model = calibration_package['calibration_model']
    calibration_method = calibration_package['calibration_method']
    
    # Get raw predictions from original model
    raw_probs = original_model.predict_proba(X_scaled)[:, 1]
    
    # Apply calibration
    if calibration_method.lower() == 'isotonic':
        calibrated_probs = calibration_model.predict(raw_probs)
    else:  # platt scaling
        calibrated_probs = calibration_model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    
    return calibrated_probs

if __name__ == "__main__":
    ticker = input("Enter ticker symbol: ")
    method = input("Calibration method (isotonic/platt): ") or "isotonic"
    calibrate_model(ticker, method)
