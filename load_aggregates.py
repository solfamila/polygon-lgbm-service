import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv
from datetime import datetime

# --- Configuration ---
load_dotenv() # Load environment variables from .env file (primarily for DB password)

CSV_FILENAME = "TSLA_minute_2025-04-12.csv" # The file generated earlier
DB_TABLE_NAME = "stock_aggregates_min"
# Match these with your docker-compose.yml `timescaledb` environment
DB_NAME = "polygondata"
DB_USER = "polygonuser"
DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass") # Load from .env or use default
# Since the script runs on the HOST, connect to the HOST port mapped to the DB container
DB_HOST = "localhost"
DB_PORT = "5433" # The port mapped on the host in docker-compose.yml

BATCH_SIZE = 1000 # Insert rows in batches for efficiency

# --- Connect to Database ---
conn = None
try:
    print(f"Connecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    print("Connection successful.")
    cur = conn.cursor()

    # --- Read CSV Data ---
    print(f"Reading CSV file: {CSV_FILENAME}...")
    try:
        df = pd.read_csv(CSV_FILENAME)
        # Make sure timestamp is in datetime format if not already
        if 'timestamp' not in df.columns and 'timestamp_ms' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')

        print(f"Read {len(df)} rows from CSV.")

        # Add symbol column needed for DB table
        df['symbol'] = 'TSLA' 
         # Handle potential missing vwap/num_trades (fill with None/NaN which becomes NULL in DB)
        if 'vwap' not in df.columns: df['vwap'] = pd.NA
        if 'num_trades' not in df.columns: df['num_trades'] = pd.NA


    except FileNotFoundError:
        print(f"Error: CSV file not found: {CSV_FILENAME}")
        exit(1)
    except Exception as e:
         print(f"Error reading CSV: {e}")
         exit(1)


    # Prepare data for insertion (ensure correct order and handle NaT/NaN)
    # Match the order of columns in the INSERT statement precisely
    # Assuming table columns: symbol, agg_open, agg_high, agg_low, agg_close, volume, vwap, startTime, endTime, numTrades
    # Note: startTime is mandatory, endTime is derived from startTime+1min usually for minute aggs
    # Map DataFrame columns to table columns
    
    # --- Prepare data for insertion --- (Modified Loop)
    data_to_insert = []
    print("Preparing data for batch insertion...")
    skipped_rows = 0
    for index, row in df.iterrows():
        # Ensure timestamp is parsed correctly
        startTime = row['timestamp'] 
        if pd.isna(startTime):
            # print(f"Skipping row {index+2} due to invalid startTime.") # Uncomment for debugging
            skipped_rows += 1
            continue # Skip this row if timestamp is invalid

        # Ensure startTime is a datetime object before adding Timedelta
        if not isinstance(startTime, pd.Timestamp):
            # Try coercing if it's not already a Timestamp (should be rare after read_csv parsing)
            try:
                startTime = pd.to_datetime(startTime)
                if pd.isna(startTime): raise ValueError("Conversion failed")
            except Exception:
                # print(f"Skipping row {index+2} due to non-datetime startTime: {startTime}") # Uncomment for debugging
                skipped_rows += 1
                continue

        # Calculate endTime safely
        endTime = startTime + pd.Timedelta(minutes=1)

        # Handle potential NaN/NA in numeric fields -> map to None (SQL NULL)
        vwap_val = row['vwap'] if 'vwap' in row and pd.notna(row['vwap']) else None
        num_trades_val = int(row['num_trades']) if 'num_trades' in row and pd.notna(row['num_trades']) else None
        volume_val = int(row['volume']) if 'volume' in row and pd.notna(row['volume']) else 0

        # Append the tuple
        data_to_insert.append((
            row['symbol'],          # symbol (string)
            row['open'],            # agg_open (float)
            row['high'],            # agg_high (float)
            row['low'],             # agg_low (float)
            row['close'],           # agg_close (float)
            volume_val,             # volume (int/bigint)
            vwap_val,               # vwap (float or None)
            startTime,              # startTime (datetime)
            endTime,                # endTime (datetime)
            num_trades_val          # numTrades (int or None)
        ))
    
    if skipped_rows > 0:
        print(f"Skipped {skipped_rows} rows due to invalid startTime.")

    # --- Insert Data in Batches ---
    print(f"Starting batch insert (Batch Size: {BATCH_SIZE})...")
    # IMPORTANT: Adjust column names to match your ACTUAL table structure from StockAggregate entity
    sql_insert = f"""
        INSERT INTO {DB_TABLE_NAME} (
            symbol, agg_open, agg_high, agg_low, agg_close, 
            volume, vwap, start_time, end_time, num_trades
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING; 
    """
    # ON CONFLICT DO NOTHING prevents errors if you run the script twice with the same data

    execute_batch(cur, sql_insert, data_to_insert, page_size=BATCH_SIZE)

    conn.commit() # Commit the transaction
    print(f"Successfully inserted/updated {len(data_to_insert)} rows into {DB_TABLE_NAME}.")


except psycopg2.Error as e:
    print(f"Database error: {e}")
    if conn:
        conn.rollback() # Roll back on error
except Exception as e:
     print(f"An error occurred: {e}")
finally:
    if cur:
        cur.close()
    if conn:
        conn.close()
        print("Database connection closed.")
