# --- START OF FILE paper_trading.py ---

# --- Necessary Imports ---
import psycopg2
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import threading # Only if running other logic in parallel
import traceback # For detailed error printing

print("--- Paper Trading Simulation ---")

# --- Configuration & DB Connection ---
load_dotenv(); DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST="localhost"; DB_PORT="5433"; DB_NAME="polygondata"; DB_USER="polygonuser"
TICKER_TO_TRADE = "TSLA" # Focus on one ticker

# --- Paper Trading Parameters ---
INITIAL_CAPITAL = 200000.00 # Use float for currency
available_capital = INITIAL_CAPITAL
MAX_SHARES_PER_TRADE = 50
MIN_CONFIDENCE = 0.60 # Min prediction probability to trigger trade (Adjust based on analysis!)
POSITION_TIMEOUT_HOURS = 4 # Close position automatically after this duration

# --- Constants for Target/Stop ---
TARGET_GAIN = 1.0 # Assumed gain used by prediction service
STOP_LOSS_DIVISOR = 2.0 # e.g. Stop is half the target gain below entry

# --- In-memory Store for Open Positions ---
# Structure: { 'TICKER': { 'trade_id': int, 'entry_time': datetime, 'entry_price': float,
#                           'shares': int, 'target_price': float, 'stop_loss': float, 'confidence': float } }
open_positions = {}

# --- Database Connection & Setup ---
conn = None
cursor = None
try:
    print(f"Connecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connection successful.")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS paper_trades (
        id SERIAL PRIMARY KEY, entry_time TIMESTAMPTZ, exit_time TIMESTAMPTZ,
        ticker VARCHAR(10), entry_price NUMERIC(12, 4), exit_price NUMERIC(12, 4),
        shares INTEGER, profit_loss NUMERIC(12, 4), trade_status VARCHAR(20),
        confidence NUMERIC(10, 4) );
    """)
    print("Paper trades table verified/created.")

except Exception as e: print(f"FATAL: DB setup failed: {e}"); exit(1)

# --- Trading Logic Functions ---
def check_for_signals():
    """ Checks stock_predictions table for actionable BUY signals. """
    global available_capital, open_positions, cursor # Declare needed globals

    # Use NOW() AT TIME ZONE 'UTC' for proper TZ comparison
    query = """
    SELECT timestamp, ticker, current_price, predicted_probability, target_price, stop_loss
    FROM stock_predictions
    WHERE ticker = %(ticker)s AND trade_signal = true
      AND timestamp > (NOW() AT TIME ZONE 'UTC' - INTERVAL '5 minutes')
    ORDER BY timestamp DESC LIMIT 1;
    """
    try:
        if cursor is None or cursor.closed: raise ConnectionError("DB Cursor closed")
        cursor.execute(query, {'ticker': TICKER_TO_TRADE}); signal = cursor.fetchone()
    except Exception as e: print(f"DB Error fetching signals: {e}"); return

    if not signal: return

    timestamp, ticker, entry_price, confidence, target_price, stop_loss = signal
    try: # Ensure data types are correct
        entry_price = float(entry_price)
        confidence = float(confidence)
        # Provide defaults if target/stop are None from prediction service
        target_price = float(target_price) if target_price is not None else entry_price + TARGET_GAIN
        stop_loss = float(stop_loss) if stop_loss is not None else entry_price - (TARGET_GAIN / STOP_LOSS_DIVISOR)
    except Exception as e: print(f"Error converting signal data types: {e}"); return

    # --- Pre-Trade Checks ---
    if confidence < MIN_CONFIDENCE:
        # print(f"Signal confidence {confidence:.4f} < {MIN_CONFIDENCE}. Skipping.") # Reduce verbosity
        return
    if ticker in open_positions:
        print(f"Signal received for {ticker}, but position already open. Skipping.")
        return
    if available_capital < entry_price: print(f"Insufficient capital: Have ${available_capital:.2f}, Need ${entry_price:.2f} for 1 share"); return

    # --- Position Sizing ---
    capital_to_risk = available_capital * 0.10 # Risk 10% of available capital
    shares = min(int(capital_to_risk / entry_price), MAX_SHARES_PER_TRADE)
    if shares <= 0: print(f"Calculated 0 shares based on risk %. Skipping."); return
    position_cost = shares * entry_price
    if position_cost > available_capital:
        shares = int(available_capital / entry_price) # Max affordable shares if risk % exceeds capital
        position_cost = shares * entry_price
    if shares <= 0: print(f"Could not afford even 1 share after adjustment."); return

    # --- Execute Buy (Logically) ---
    print(f"\nReceived BUY Signal: {ticker} @ ${entry_price:.4f}, Conf: {confidence:.4f}, Time: {timestamp}")
    available_capital -= position_cost

    # --- Log to DB ---
    insert_sql = """INSERT INTO paper_trades (entry_time, ticker, entry_price, shares, trade_status, confidence) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;"""
    trade_values = (timestamp, ticker, entry_price, shares, 'OPEN', confidence)
    trade_id = None
    try:
        if cursor is None or cursor.closed: raise ConnectionError("DB Cursor closed")
        cursor.execute(insert_sql, trade_values); trade_id = cursor.fetchone()[0]; print(f"Trade {trade_id} logged to DB.")
    except Exception as e: print(f"DB Error logging OPEN trade: {e}"); available_capital += position_cost; return # Rollback capital subtraction

    # --- Store in Memory ---
    open_positions[ticker] = { 'trade_id': trade_id, 'entry_time': timestamp, 'ticker': ticker, 'entry_price': entry_price, 'shares': shares, 'target_price': target_price, 'stop_loss': stop_loss, 'confidence': confidence }
    print(f"--- PAPER TRADE: OPENED --- | Ticker: {ticker} | Shares: {shares} | Entry: ${entry_price:.2f} | Cost: ${position_cost:.2f} | Target: ${target_price:.2f} | Stop: ${stop_loss:.2f} | Cash Left: ${available_capital:.2f}")


def check_open_positions():
    """ Checks if open positions should be closed based on price or timeout. """
    global available_capital, open_positions, cursor # Declare needed globals
    if not open_positions: return

    tickers_tuple = tuple(open_positions.keys())
    if not tickers_tuple: return # Should not happen if open_positions is not empty, but safe check

    # <<< FIX: Use IN operator instead of ANY >>>
    query = """
    WITH RankedAggs AS (
        SELECT symbol, agg_close, start_time,
               ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY start_time DESC) as rn
        FROM stock_aggregates_min
        WHERE symbol IN %s -- Use IN operator
    )
    SELECT symbol, agg_close, start_time
    FROM RankedAggs
    WHERE rn = 1;
    """

    latest_prices = {}; latest_timestamps = {}
    try:
        if cursor is None or cursor.closed: raise ConnectionError("DB Cursor closed")
        # Pass the tuple within another tuple for parameter substitution with IN
        cursor.execute(query, (tickers_tuple,))
        results = cursor.fetchall()
        # Ensure results format is as expected (ticker, price, timestamp)
        latest_prices = {res[0]: float(res[1]) for res in results}
        latest_timestamps = {res[0]: res[2] for res in results}
    except psycopg2.Error as e: print(f"DB Error fetching prices for open positions: {e}"); return
    except IndexError: print(f"Error: Unexpected result format from price query: {results}"); return # Catch if result structure is wrong
    except Exception as e: print(f"Error fetching/processing prices: {e}"); return

    current_time_utc = datetime.now(timezone.utc)

    # Iterate over a copy of the keys because we might delete items from the dict
    for ticker in list(open_positions.keys()):
        position = open_positions[ticker]

        # Check if we got a price for this specific ticker
        if ticker not in latest_prices:
            print(f"Warn: No current price data fetched for open position {ticker}. Skipping check for this cycle.")
            continue # Skip this ticker if no price was found

        current_price = latest_prices[ticker]
        exit_reason = None

        # Ensure entry time is timezone-aware for comparison
        entry_time_utc = position['entry_time']
        if entry_time_utc.tzinfo is None: # If somehow entry_time is naive, assume UTC
            print(f"Warn: Position {position['trade_id']} entry time was naive, assuming UTC.")
            entry_time_utc = entry_time_utc.replace(tzinfo=timezone.utc)

        position_age = current_time_utc - entry_time_utc

        # --- Check Exit Conditions ---
        if current_price >= position['target_price']: exit_reason = 'TARGET_HIT'
        elif current_price <= position['stop_loss']: exit_reason = 'STOP_LOSS'
        elif position_age > timedelta(hours=POSITION_TIMEOUT_HOURS): exit_reason = 'TIMEOUT'

        if exit_reason:
            print(f"\n--- PAPER TRADE: CLOSING {position['trade_id']} ({ticker} {exit_reason}) ---")
            shares=position['shares']; entry_price=position['entry_price']; exit_price=current_price
            profit_loss = shares * (exit_price - entry_price); capital_return = shares * exit_price

            # --- Update DB ---
            # Use the timestamp associated with the fetched price, or now() if missing
            exit_timestamp = latest_timestamps.get(ticker, datetime.now(timezone.utc))
            if exit_timestamp.tzinfo is None: # Ensure timezone-aware
                 exit_timestamp = exit_timestamp.replace(tzinfo=timezone.utc)

            update_sql = """UPDATE paper_trades SET exit_time = %s, exit_price = %s, profit_loss = %s, trade_status = %s WHERE id = %s AND trade_status = 'OPEN'; """
            update_values = (exit_timestamp, exit_price, profit_loss, exit_reason, position['trade_id'])
            try:
                if cursor is None or cursor.closed: raise ConnectionError("DB Cursor closed")
                cursor.execute(update_sql, update_values)
                rows_updated = cursor.rowcount # Check if the update actually happened
                if rows_updated > 0:
                    # --- Update State ONLY AFTER successful DB update ---
                    available_capital += capital_return
                    print(f"  Trade {position['trade_id']} updated in DB.")
                    print(f"  Exit: ${exit_price:.2f} at {exit_timestamp} | P/L: ${profit_loss:.2f} ({(profit_loss / (shares * entry_price) * 100) if shares*entry_price != 0 else 0:.2f}%) | Held: {position_age}")
                    print(f"  Current Cash: ${available_capital:.2f}")
                    del open_positions[ticker] # Remove from memory ONLY if DB updated
                else:
                     # This can happen if another process closed it, or if the status wasn't 'OPEN'
                     print(f"  Warn: Trade {position['trade_id']} ({ticker}) DB update affected 0 rows. Was it already closed or status != 'OPEN'?")
                     # If the trade is no longer in DB as OPEN, remove from memory too
                     if ticker in open_positions: # Double check it's still there
                         del open_positions[ticker]
                         print(f"  Removed {ticker} from open positions memory as DB update failed/skipped.")


            except Exception as e:
                print(f"  DB Error during trade {position['trade_id']} ({ticker}) CLOSE update: {e}")
                # Do NOT remove from open_positions here, as the DB state is uncertain. Retry next cycle.

def show_portfolio_status():
    """ Calculates and prints current portfolio status. """
    global available_capital, open_positions, cursor, INITIAL_CAPITAL

    portfolio_value = available_capital; open_positions_value = 0.0
    latest_prices = {} # Define upfront

    if open_positions:
        tickers_tuple = tuple(open_positions.keys())
        if tickers_tuple: # Check if tuple is not empty
             # <<< FIX: Use IN operator instead of ANY >>>
             query = """ WITH RankedAggs AS (SELECT symbol, agg_close, ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY start_time DESC) as rn FROM stock_aggregates_min WHERE symbol IN %s) SELECT symbol, agg_close FROM RankedAggs WHERE rn = 1; """
             try:
                 if cursor is None or cursor.closed: raise ConnectionError("DB Cursor closed")
                 # Pass the tuple within another tuple for parameter substitution with IN
                 cursor.execute(query, (tickers_tuple,))
                 # Expecting (ticker, price) pairs
                 results = cursor.fetchall()
                 latest_prices = {res[0]: float(res[1]) for res in results}
             except psycopg2.Error as e: print(f" DB Warn: Could not get prices for status query: {e}") # Keep warning
             except IndexError: print(f"Error: Unexpected result format from status price query: {results}")
             except Exception as e: print(f" Error getting/processing prices for status: {e}")

             for ticker, position in open_positions.items():
                 # Use entry price as fallback if current price fetch failed
                 current_price = latest_prices.get(ticker, position['entry_price'])
                 position_value = position['shares'] * current_price
                 open_positions_value += position_value

    portfolio_value += open_positions_value

    # Get closed trade performance
    total_trades, winning_trades, total_pnl = 0, 0, 0.0
    try:
        if cursor is None or cursor.closed: raise ConnectionError("DB Cursor closed")
        # Ensure we only count trades that are actually finished
        cursor.execute("""SELECT COALESCE(COUNT(*), 0),
                                 COALESCE(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END), 0),
                                 COALESCE(SUM(profit_loss), 0.0)
                          FROM paper_trades
                          WHERE trade_status NOT IN ('OPEN', 'PENDING_ENTRY');""") # Exclude OPEN and PENDING
        result = cursor.fetchone();
        if result:
            total_trades = result[0] if result[0] is not None else 0
            winning_trades = result[1] if result[1] is not None else 0
            total_pnl = float(result[2]) if result[2] is not None else 0.0 # Handle None sum explicitly
    except Exception as e: print(f" DB Error fetching trade stats: {e}")

    # --- Print Summary ---
    print("\n" + "="*25 + " PORTFOLIO STATUS " + "="*25)
    print(f"Timestamp:               {datetime.now(timezone.utc)}")
    print(f"Initial Capital:         ${INITIAL_CAPITAL:.2f}")
    print(f"Current Cash:            ${available_capital:.2f}")
    print(f"Value of Open Positions: ${open_positions_value:.2f}")
    print(f"Total Portfolio Value:   ${portfolio_value:.2f}")
    performance_pct = ((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL else 0
    print(f"Performance:             {performance_pct:+.2f}%") # Added sign
    print(f"Open Positions Count:    {len(open_positions)}")
    if open_positions:
        print("  --- Open Position Details ---")
        for ticker, pos in open_positions.items():
            current_price = latest_prices.get(ticker, pos['entry_price']) # Get price again for display
            current_val = pos['shares'] * current_price
            unrealized_pnl = current_val - (pos['shares'] * pos['entry_price'])
            print(f"    {ticker}: {pos['shares']} @ ${pos['entry_price']:.2f} -> ${current_price:.2f} (Target: ${pos['target_price']:.2f}, Stop: ${pos['stop_loss']:.2f}) | Val: ${current_val:.2f} | UPL: ${unrealized_pnl:+.2f}") # Added sign
    print("-" * 25); print("Closed Trades Summary:")
    print(f"  Total Closed:          {total_trades}")
    if total_trades > 0:
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        print(f"  Wins / Losses:         {winning_trades} / {total_trades - winning_trades}")
        print(f"  Win Rate (Closed):     {win_rate:.2f}%")
        print(f"  Total P&L (Closed):    ${total_pnl:+.2f}") # Added sign
        print(f"  Average P&L / Trade:   ${avg_pnl:+.2f}") # Added sign
    print("="*68)

# ==============================================================================
#                             MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == "__main__":
    print("\nStarting paper trading system...")
    print(f"Trading Ticker: {TICKER_TO_TRADE}")
    print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Confidence Threshold: {MIN_CONFIDENCE}")
    print(f"Position Timeout: {POSITION_TIMEOUT_HOURS} hours")

    try:
        status_report_interval = 60 # Seconds (1 minute for more frequent updates during testing)
        last_status_report_time = time.time() - status_report_interval # Report immediately on start

        while True:
            loop_start_time = time.time()

            # Recheck DB connection (simple check)
            if conn is None or conn.closed:
                print("DB connection lost. Attempting reconnect...")
                try:
                    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
                    conn.autocommit = True
                    cursor = conn.cursor()
                    print("DB Reconnection successful.")
                except Exception as e:
                    print(f"FATAL: DB Reconnect failed: {e}. Sleeping before retry...")
                    time.sleep(60) # Wait before retrying connection
                    continue # Skip rest of loop

            # Perform trading checks
            check_for_signals()      # Check for new BUY signals
            check_open_positions()   # Check if existing positions hit T/S or timeout

            # Status Reporting
            current_time = time.time()
            if current_time - last_status_report_time >= status_report_interval:
                show_portfolio_status() # Show status periodically
                last_status_report_time = current_time

            # Loop Timing Control
            sleep_duration = 30 # Check every 30 seconds
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, sleep_duration - elapsed_time) # Prevent negative sleep
            # print(f"Loop took {elapsed_time:.2f}s, sleeping for {sleep_time:.2f}s...") # Optional debug print
            time.sleep(sleep_time)

    except KeyboardInterrupt: print("\nPaper trading system stopped by user.")
    except Exception as e: print(f"\nFATAL ERROR in main loop: {e}"); traceback.print_exc()
    finally:
        print("\n--- Shutting down ---")
        if cursor: cursor.close()
        if conn: conn.close(); print("Database connection closed.")
        print("\n--- FINAL Portfolio Status ---")
        try:
            # Need a temporary connection for final status if main one closed
            temp_conn = None
            temp_cursor = None
            try:
                temp_conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
                temp_cursor = temp_conn.cursor()
                cursor = temp_cursor # Temporarily assign for show_portfolio_status
                show_portfolio_status() # Show status one last time
            except Exception as final_e:
                print(f"Could not fetch final status: {final_e}")
            finally:
                if temp_cursor: temp_cursor.close()
                if temp_conn: temp_conn.close()
        except NameError: # If cursor wasn't defined due to initial connection failure
             print("Initial DB connection failed, cannot show final status.")

        print("Exiting paper trading simulation.")

# --- END OF FILE paper_trading.py ---
