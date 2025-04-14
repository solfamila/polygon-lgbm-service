import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

API_KEY = os.getenv("POLYGON_API_KEY")
TICKER = "TSLA"
# Fetch data for the last N weeks (adjust as needed)
WEEKS_AGO = 3
# Aggregate details
MULTIPLIER = 1
TIMESPAN = "minute" # minute, hour, day, week, month, quarter, year
# Output file name
CSV_FILENAME_TEMPLATE = "{ticker}_{timespan}_{end_date}.csv"

# --- Date Calculation ---
end_date = datetime.today().date()
start_date = end_date - timedelta(weeks=WEEKS_AGO)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# --- Input Validation ---
if not API_KEY:
    print("Error: POLYGON_API_KEY environment variable not set.")
    print("Please create a .env file with POLYGON_API_KEY=YOUR_KEY")
    exit(1)

print(f"Fetching {MULTIPLIER} {TIMESPAN} aggregates for {TICKER}")
print(f"Date range: {start_date_str} to {end_date_str}")

# --- API Call Setup ---
# Using v2 Aggregates API: https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksTickers__range__multiplier___timespan___from___to
base_url = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/{MULTIPLIER}/{TIMESPAN}/{start_date_str}/{end_date_str}"
params = {
    "adjusted": "true", # Typically true for adjusted prices
    "sort": "asc",     # Sort by time ascending
    "limit": 50000,    # Max limit per request
    "apiKey": API_KEY  # Add API key as a parameter
}

all_results = []
url = base_url # Start with the base URL

# --- Fetching Loop (Handles Pagination) ---
page_count = 0
while url:
    page_count += 1
    print(f"Fetching page {page_count}...")
    try:
        response = requests.get(url, params=params if page_count == 1 else None) # Params only needed for the first request if using next_url
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        if 'results' in data and data['results']:
            print(f"  -> Fetched {len(data['results'])} results.")
            all_results.extend(data['results'])
        else:
            print("  -> No more results found.")
            break # Exit loop if no results

        # --- Pagination ---
        # Polygon's v2 API often uses 'next_url' for pagination
        # This 'next_url' already includes the API key and cursor parameters
        if 'next_url' in data:
            url = data['next_url']
            # Remove standard params if using next_url, as it contains everything needed
            params = None 
            print("  -> Following next_url...")
        else:
            print("  -> No next_url found, fetch complete.")
            url = None # End the loop

        # --- Rate Limiting ---
        # Free plan limit is 5 calls/min. Paid plans are higher but politeness helps.
        # Add a small delay (adjust if needed based on your subscription level)
        time.sleep(0.2) # 200ms delay, adjust if hitting limits

    except requests.exceptions.RequestException as e:
        print(f"\nError during API request: {e}")
        # You might want to implement retries here for robustness
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        break

# --- Data Processing & Saving ---
if all_results:
    print(f"\nTotal bars fetched: {len(all_results)}")

    # Convert to Pandas DataFrame
    df = pd.DataFrame(all_results)

    # Rename columns for clarity and convention (Polygon uses single letters)
    # 't' = timestamp (ms), 'o' = open, 'h' = high, 'l' = low, 'c' = close, 'v' = volume, 'vw' = vwap, 'n' = num_trades
    column_map = {
        't': 'timestamp_ms',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vw': 'vwap',
        'n': 'num_trades'
    }
    df = df.rename(columns=column_map)

    # Convert millisecond timestamp to datetime objects (optional but recommended)
    # Using errors='coerce' will turn unparseable timestamps into NaT (Not a Time)
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')

    # Select and reorder columns (optional)
    columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades', 'timestamp_ms']
    # Filter out columns that might not exist if they weren't in the API response
    df_final = df[[col for col in columns_to_keep if col in df.columns]]

    # Save to CSV
    filename = CSV_FILENAME_TEMPLATE.format(ticker=TICKER, timespan=TIMESPAN, end_date=end_date_str)
    try:
        df_final.to_csv(filename, index=False)
        print(f"Data successfully saved to: {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

else:
    print("\nNo data was fetched or processed.")
