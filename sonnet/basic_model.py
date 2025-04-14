import psycopg2
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib # For saving the model

# --- Configuration & DB Connection ---
load_dotenv() 
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "polygondata"
DB_USER = "polygonuser"
TICKER = "TSLA"

conn = None
try:
    print(f"Connecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    print("Connection successful.")

    # --- Fetch Historical Data ---
    print(f"Fetching historical data for {TICKER}...")
    query = """
    SELECT 
        start_time AS time, 
        agg_open   AS open,  
        agg_high   AS high,  
        agg_low    AS low,   
        agg_close  AS close, 
        volume               
    FROM 
        stock_aggregates_min 
    WHERE 
        symbol = %(ticker)s -- Use parameter binding
    ORDER BY 
        start_time ASC;   
    """
    # Use pandas read_sql with parameters for safety
    df = pd.read_sql(query, conn, params={'ticker': TICKER})
    
    # Set the timestamp column as the DataFrame index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    print(f"Loaded {len(df)} historical bars for {TICKER}")

except psycopg2.Error as e:
    print(f"Database error: {e}")
    exit(1) # Exit if data loading fails
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit(1) # Exit if data loading fails
finally:
    if conn:
        conn.close()
        print("Database connection closed.")
        
# Exit if no data was loaded
if df.empty:
    print("No data loaded from database. Exiting.")
    exit(1)

# --- Basic Feature Engineering ---
print("Creating features...")
def create_basic_features(df):
    # Make sure DataFrame has enough rows 
    min_required_rows = 20 + 15 # Max window + lookahead period
    if len(df) < min_required_rows:
         print(f"Warning: DataFrame has {len(df)} rows, need at least {min_required_rows} for feature engineering.")
         return pd.DataFrame() 

    df = df.copy()
    
    # Price features
    df['return'] = df['close'].pct_change()
    df['high_low_range'] = df['high'] - df['low']
    
    # Moving averages
    for window in [5, 20]:
        df[f'ma_{window}'] = df['close'].rolling(window=window, min_periods=window).mean()
    
    # Momentum
    for window in [5, 10]:
        df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
    
    # Target: Will price increase by $1 within next 15 minutes?
    df['future_price'] = df['close'].shift(-15)
    df['target'] = (df['future_price'] - df['close'] >= 1).astype(int)
    
    # Drop rows with NaNs created by feature engineering 
    df.dropna(inplace=True)
    
    return df

feature_df = create_basic_features(df)

# Exit if feature engineering failed or produced no rows
if feature_df.empty:
    print("Feature DataFrame is empty after dropping NaNs. Exiting.")
    exit(1)

print(f"Data after feature engineering: {len(feature_df)} rows")
print(f"Number of $1 gain opportunities: {feature_df['target'].sum()} ({feature_df['target'].mean()*100:.2f}%)")

# --- Build and Test Basic Model ---

# Prepare features and target for ML
# Make sure to drop columns not usable as features
columns_to_drop = ['target', 'future_price', 'open', 'high', 'low'] # Also drop raw OHLC maybe? depends on strategy
X = feature_df.drop(columns=columns_to_drop) 
y = feature_df['target']

# Check if features or target are empty after dropping
if X.empty or y.empty:
    print("Error: Feature set (X) or target (y) is empty after dropping columns. Check feature engineering.")
    exit(1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False) # Important: shuffle=False for time series!

# Verify split results
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
if X_train.empty or X_test.empty:
    print("Error: Train or test split resulted in empty data. Check data size and split ratio.")
    exit(1)


# Normalize features
# Handle potential all-NaN columns after split (unlikely but possible)
X_train = X_train.dropna(axis=1, how='all')
X_test = X_test[X_train.columns] # Keep only columns present in Train after dropna
X_test = X_test.fillna(X_test.mean()) # Basic imputation for any remaining test NaNs using test mean

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple RandomForest model first
print("Training RandomForest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1) # Use balanced weights, utilize all cores
rf_model.fit(X_train_scaled, y_train)

# Evaluate
print("Evaluating model...")
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nModel Performance (RandomForest):")
print(classification_report(y_test, y_pred, zero_division=0))

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (RandomForest)')
plt.colorbar()
tick_marks = np.arange(len(set(y)))
plt.xticks(tick_marks, set(y), rotation=45)
plt.yticks(tick_marks, set(y))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
cm_filename = 'rf_confusion_matrix.png'
try:
    plt.savefig(cm_filename)
    print(f"\nConfusion matrix saved to {cm_filename}")
except Exception as e:
    print(f"Error saving confusion matrix plot: {e}")
plt.close()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns, # Use columns from X_train after potential dropna
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Top 10):")
print(feature_importance.head(10))

# Save the model and scaler
model_filename = 'randomforest_basic_model.joblib'
try:
    joblib.dump((rf_model, scaler, X_train.columns), model_filename) # Save columns too
    print(f"Model, scaler, and columns saved to {model_filename}")
except Exception as e:
    print(f"Error saving model: {e}")
