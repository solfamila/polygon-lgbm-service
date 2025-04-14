# --- Necessary Imports ---
import psycopg2
import pandas as pd
import numpy as np # Used indirectly via pandas
import time
import os
from datetime import datetime, timedelta, timezone 
from dotenv import load_dotenv
import threading # Only if running prediction in same script, not needed here

print("--- Paper Trading Simulation ---")

# --- Configuration & DB Connection ---
load_dotenv(); DB_PASS = os.getenv("POLYGON_DB_PASSWORD", "polygonpass")
DB_HOST="localhost"; DB_PORT="5433"; DB_NAME="polygondata"; DB_USER="polygonuser"
TICKER_TO_TRADE = "TSLA" # Focus on one ticker for now

# --- Paper Trading Parameters ---
INITIAL_CAPITAL = 10000.00
available_capital = INITIAL_CAPITAL 
MAX_SHARES_PER_TRADE = 50  
MIN_CONFIDENCE = 0.60 # Min prediction probability to trigger trade
POSITION_TIMEOUT_HOURS = 4 # Close position automatically after this duration

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
    
    # Create paper trading table if needed (corrected precision)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS paper_trades (
        id SERIAL PRIMARY KEY, entry_time TIMESTAMPTZ, exit_time TIMESTAMPTZ,
        ticker VARCHAR(10), entry_price NUMERIC(12, 4), exit_price NUMERIC(12, 4),
        shares INTEGER, profit_loss NUMERIC(12, 4), trade_status VARCHAR(20), 
        confidence NUMERIC(10, 4) );
    """)
    print("Paper trades table verified/created.")
    # Optional: Clean up any 'OPEN' trades from previous interrupted runs?
    # cursor.execute("DELETE FROM paper_trades WHERE trade_status = 'OPEN';") 
    # print("Cleared any lingering OPEN trades from previous runs.")

except psycopg2.Error as e: print(f"FATAL: DB connection/setup failed: {e}"); exit(1)
except Exception as e: print(f"FATAL: Setup error: {e}"); exit(1)

# --- Trading Logic Functions ---
def check_for_signals():
    """ Checks stock_predictions table for actionable BUY signals. """
    global available_capital, open_positions, cursor

    query = """
    SELECT timestamp, ticker, current_price, predicted_probability, target_price, stop_loss
    FROM stock_predictions
    WHERE ticker = %(ticker)s AND trade_signal = true 
      AND timestamp > (NOW() AT TIME ZONE 'UTC' - INTERVAL '5 minutes') 
    ORDER BY timestamp DESC LIMIT 1; 
    """
    try:
        cursor.execute(query, {'ticker': TICKER_TO_TRADE})
        signal = cursor.fetchone()
    except psycopg2.Error as e: print(f"DB Error fetching signals: {e}"); return
    except Exception as e: print(f"Error fetching signals: {e}"); return
        
    if not signal: return # No recent BUY signals found
    
    # Unpack signal data
    timestamp, ticker, entry_price, confidence, target_price, stop_loss = signal
    entry_price = float(entry_price) # Ensure numeric types
    confidence = float(confidence)
    # Handle potential None for target/stop if prediction service saves NULL
    target_price = float(target_price) if target_price else entry_price + TARGET_GAIN # Recalculate target if needed
    stop_loss = float(stop_loss) if stop_loss else entry_price - (TARGET_GAIN / 2) # Recalculate stop if needed
    
    # --- Pre-Trade Checks ---
    if confidence < MIN_CONFIDENCE: return # Skip low confidence
    if ticker in open_positions: return    # Skip if already holding
    if available_capital < entry_price: print(f"Insufficient capital: Have ${available_capital:.2f}, Need ${entry_price:.2f}"); return

    # --- Position Sizing ---
    capital_to_risk = available_capital * 0.10 # Example: Risk 10%
    shares = min(int(capital_to_risk / entry_price), MAX_SHARES_PER_TRADE)
    if shares <= 0: print(f"Calculated 0 shares to trade."); return
    position_cost = shares * entry_price
    if position_cost > available_capital: print("Adjusting shares due to cost exceeding capital."); shares = int(available_capital / entry_price); position_cost = shares * entry_price
    if shares <= 0: return

    # --- Execute Buy (Logically) ---
    print(f"\nReceived BUY Signal: {ticker} @ ${entry_price:.4f}, Conf: {confidence:.4f}, Time: {timestamp}")
    available_capital -= position_cost 
    
    # --- Log to DB ---
    insert_sql = """INSERT INTO paper_trades (entry_time, ticker, entry_price, shares, trade_status, confidence) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;"""
    trade_values = (timestamp, ticker, entry_price, shares, 'OPEN', confidence)
    trade_id = None
    try:
        cursor.execute(insert_sql, trade_values)
        trade_id = cursor.fetchone()[0]
        print(f"Trade {trade_id} logged to DB.")
    except Exception as e: print(f"DB Error logging OPEN trade {trade_id}: {e}"); available_capital += position_cost; return # Rollback capital

    # --- Store in Memory ---
    open_positions[ticker] = { 'trade_id': trade_id, 'entry_time': timestamp, 'ticker': ticker, 'entry_price': entry_price, 'shares': shares, 'target_price': target_price, 'stop_loss': stop_loss, 'confidence': confidence }
    print(f"--- PAPER TRADE: OPENED --- | Ticker: {ticker} | Shares: {shares} | Entry: ${entry_price:.2f} | Cost: ${position_cost:.2f} | Target: ${target_price:.2f} | Stop: ${stop_loss:.2f} | Cash Left: ${available_capital:.2f}")


def check_open_positions():
    """ Checks if open positions should be closed. """
    global available_capital, open_positions, cursor
    if not open_positions: return

    tickers = tuple(open_positions.keys())
    if not tickers: return 

    # Query for latest price using correct columns
    query = """
    WITH RankedAggs AS (SELECT symbol, agg_close, start_time, ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY start_time DESC) as rn
                       FROM stock_aggregates_min WHERE symbol = ANY(%(tickers)s))
    SELECT symbol, agg_close, start_time FROM RankedAggs WHERE rn = 1;"""
    
    latest_prices = {}; latest_timestamps = {}
    try: cursor.execute(query, {'tickers': tickers}); results = cursor.fetchall(); latest_prices = {t: float(p) for t,p,ts in results}; latest_timestamps = {t: ts for t,p,ts in results}
    except Exception as e: print(f"DB Error fetching prices for open positions: {e}"); return

    current_time_utc = datetime.now(timezone.utc) 

    for ticker in list(open_positions.keys()):
        position = open_positions[ticker]
        if ticker not in latest_prices: print(f"Warn: No current price for {ticker}, skipping close check."); continue
        current_price = latest_prices[ticker]
        
        # --- Check Exit Conditions ---
        exit_reason = None
        position_age = current_time_utc - position['entry_time'] # Timezone aware comparison

        if current_price >= position['target_price']: exit_reason = 'TARGET_HIT'
        elif current_price <= position['stop_loss']: exit_reason = 'STOP_LOSS'
        elif position_age > timedelta(hours=POSITION_TIMEOUT_HOURS): exit_reason = 'TIMEOUT'
            
        # --- Execute Close ---
        if exit_reason:
            print(f"\n--- PAPER TRADE: CLOSING {ticker} ({exit_reason}) ---")
            shares = position['shares']; entry_price = position['entry_price']; exit_price = current_price
            profit_loss = shares * (exit_price - entry_price)
            capital_return = shares * exit_price
            
            # --- Update DB ---
            update_sql = """UPDATE paper_trades SET exit_time = (NOW() AT TIME ZONE 'UTC'), exit_price = %s, profit_loss = %s, trade_status = %s WHERE id = %s AND trade_status = 'OPEN'; """
            update_values = (exit_price, profit_loss, exit_reason, position['trade_id'])
            try:
                cursor.execute(update_sql, update_values)
                print(f"  Trade {position['trade_id']} updated in DB.")
                
                # --- Update State ONLY AFTER successful DB update ---
                available_capital += capital_return # Update capital first
                print(f"  Exit: ${exit_price:.2f} | P/L: ${profit_loss:.2f} ({(profit_loss / (shares * entry_price) * 100):.2f}%) | Held: {position_age}")
                print(f"  Current Cash: ${available_capital:.2f}")
                del open_positions[ticker] # Remove from memory

            except Exception as e: print(f"  DB Error closing trade {position['trade_id']}: {e}")


def show_portfolio_status():
    """ Calculates and prints current portfolio status. """
    global available_capital, open_positions, cursor, INITIAL_CAPITAL
    
    print("\n--- Calculating Portfolio Status ---")
    portfolio_value = available_capital 
    open_positions_value = 0.0
    
    if open_positions:
        tickers = tuple(open_positions.keys())
        # Get latest prices for open positions
        query = """ WITH RankedAggs AS (SELECT symbol, agg_close, ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY start_time DESC) as rn FROM stock_aggregates_min WHERE symbol = ANY(%(tickers)s)) SELECT symbol, agg_close FROM RankedAggs WHERE rn = 1; """
        latest_prices = {}
        try: cursor.execute(query, {'tickers': tickers}); latest_prices = {t: float(p) for t,p in cursor.fetchall()}
        except Exception as e: print(f" DB Warn: Could not get prices for status: {e}")

        for ticker, position in open_positions.items():
             current_price = latest_prices.get(ticker, position['entry_price']) # Default to entry if current price missing
             position_value = position['shares'] * current_price
             open_positions_value += position_value
             # print(f"  Open Position: {ticker} x{position['shares']} @ ${position['entry_price']:.2f} -> Value ${position_value:.2f}")

    portfolio_value += open_positions_value # Add value of holdings
    
    # Get closed trade performance
    total_trades, winning_trades, total_pnl = 0, 0, 0.0
    try:
        cursor.execute("""SELECT COALESCE(COUNT(*), 0), COALESCE(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END), 0), COALESCE(SUM(profit_loss), 0.0) FROM paper_trades WHERE trade_status <> 'OPEN';""")
        result = cursor.fetchone(); 
        if result: total_trades, winning_trades, total_pnl = result[0], result[1], float(result[2])
    except Exception as e: print(f" DB Error fetching trade stats: {e}")
    
    # --- Print Summary ---
    print("\n--- PORTFOLIO STATUS ---")
    print(f"Timestamp: {datetime.now(timezone.utc)}")
    print(f"Initial Capital:         ${INITIAL_CAPITAL:.2f}")
    print(f"Current Cash:            ${available_capital:.2f}")
    print(f"Value of Open Positions: ${open_positions_value:.2f}")
    print(f"Total Portfolio Value:   ${portfolio_value:.2f}")
    performance_pct = ((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL else 0
    print(f"Performance:             {performance_pct:.2f}%")
    print(f"Open Positions Count:    {len(open_positions)}")
    print("-" * 25); print("Closed Trades Summary:")
    print(f"  Total Closed:          {total_trades}")
    if total_trades > 0:
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        print(f"  Win Rate:              {win_rate:.2f}% ({winning_trades} wins)")
        print(f"  Total P&L (Closed):    ${total_pnl:.2f}")
    print("-" * 25)

# ==============================================================================
#                             MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == "__main__":
    print("\nStarting paper trading system...")
    print(f"Trading Ticker: {TICKER_TO_TRADE}")
    print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")

    try:
        status_report_interval = 300 # Seconds
        last_status_report_time = time.time() - status_report_interval 

        while True:
            loop_start_time = time.time()

            # 1. Check for signals from prediction service
            check_for_signals()
            
            # 2. Check/Manage open positions
            check_open_positions()
            
            # 3. Report status periodically
            current_time = time.time()
            if current_time - last_status_report_time >= status_report_interval:
                show_portfolio_status()
                last_status_report_time = current_time
            
            # Wait before next check cycle
            sleep_duration = 30 # Check every 30 seconds
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, sleep_duration - elapsed_time) 
            time.sleep(sleep_time) 
            
    except KeyboardInterrupt: print("\nPaper trading system stopped by user.")
    except Exception as e: print(f"\nFATAL ERROR in main loop: {e}"); import traceback; traceback.print_exc() 
    finally:    
        if cursor: cursor.close() # Close cursor first
        if conn: conn.close(); print("Database connection closed.")
        print("Final Portfolio Status:")
        show_portfolio_status() # Show status one last time
        print("Exiting paper trading simulation.")
