import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # <-- Make sure sklearn is installed (pip install scikit-learn)
import matplotlib.pyplot as plt                # <-- Make sure matplotlib is installed (pip install matplotlib)
import os                                       # <-- Add import for os
from dotenv import load_dotenv                  # <-- Add import for dotenv

# Load environment variables (for password, ideally)
load_dotenv()
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass") # Load password from .env or use default

# Connect to your existing TimescaleDB
conn = psycopg2.connect(
    host="localhost",
    port="5433",  # <-- USE HOST PORT 5433
    database="polygondata",  # <-- Use correct DB name
    user="polygonuser",  # <-- Use correct user
    password=DB_PASS  # <-- Use correct password (loaded from .env or default)
)

# Fetch historical data from your database
query = """
SELECT 
    start_time AS time, -- Use start_time, alias as 'time' for DataFrame
    agg_open   AS open,  -- Use agg_open, alias as 'open'
    agg_high   AS high,  -- Use agg_high, alias as 'high' 
    agg_low    AS low,   -- Use agg_low, alias as 'low' 
    agg_close  AS close, -- Use agg_close, alias as 'close'
    volume               -- Volume column name seems correct
FROM 
    stock_aggregates_min -- Correct table name
WHERE 
    symbol = 'TSLA'   -- Filter by correct symbol column name
-- GROUP BY not needed here as we're selecting raw aggregates
ORDER BY 
    start_time ASC;    -- Order by correct time column
"""

df = pd.read_sql(query, conn)
# Set the timestamp column as the DataFrame index (VERY IMPORTANT for time series analysis)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

print(f"Loaded {len(df)} historical bars for TSLA")

# Basic feature engineering (simplified version of what we'll do later)
def create_basic_features(df):
    # Make sure DataFrame has enough rows for rolling windows etc.
    if len(df) < 20: # Adjust min rows if using larger windows
         print("Warning: DataFrame too small for feature engineering.")
         return pd.DataFrame() # Return empty if not enough data
    
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
    
    # Drop rows with NaNs created by feature engineering (especially rolling/shifting)
    df.dropna(inplace=True)
    
    return df

# Close the database connection (do this *after* fetching data)
if conn:
    conn.close()
    print("Database connection closed.")

# Create features and visualize targets
if not df.empty:
    feature_df = create_basic_features(df)
    
    if not feature_df.empty:
        print(f"Data after feature engineering: {len(feature_df)} rows")
        print(f"Number of $1 gain opportunities: {feature_df['target'].sum()} ({feature_df['target'].mean()*100:.2f}%)")

        # Visualize price and target signals
        plt.figure(figsize=(15, 8))
        # Use the index (which is timestamp) for plotting
        plt.plot(feature_df.index, feature_df['close'], 'b-', alpha=0.6)
        plt.scatter(feature_df[feature_df['target'] == 1].index, 
                    feature_df[feature_df['target'] == 1]['close'],
                    color='green', label='$1 gain opportunity', s=10) # Smaller dots
        plt.title('TSLA Price Chart with $1 Gain Opportunities Highlighted')
        plt.legend()
        
        # Save plot
        plot_filename = 'tsla_opportunity_analysis.png'
        try:
            plt.savefig(plot_filename)
            print(f"Saved price chart with gain opportunities to {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close() # Close the plot window
    else:
        print("Feature DataFrame is empty after dropping NaNs.")
else:
    print("No data loaded from database.")

# Create features and visualize targets
feature_df = create_basic_features(df)
print(f"Data after feature engineering: {len(feature_df)} rows")
print(f"Number of $1 gain opportunities: {feature_df['target'].sum()} ({feature_df['target'].mean()*100:.2f}%)")

# Visualize price and target signals
plt.figure(figsize=(15, 8))
plt.plot(feature_df.index, feature_df['close'], 'b-', alpha=0.6)
plt.scatter(feature_df[feature_df['target'] == 1].index, 
            feature_df[feature_df['target'] == 1]['close'],
            color='green', label='$1 gain opportunity')
plt.title('TSLA Price Chart with $1 Gain Opportunities Highlighted')
plt.legend()
plt.savefig('opportunity_analysis.png')
plt.close()

print("Saved price chart with gain opportunities to opportunity_analysis.png")

