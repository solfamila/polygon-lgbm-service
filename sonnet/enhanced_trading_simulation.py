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
def simulate_single_day(ticker, date, params=None, fetch_missing=False):
    if params is None:
        params = DEFAULT_PARAMS
    
    # Extract parameters
    confidence_threshold = params.get('confidence_threshold', 0.57)
    target_profit_pct = params.get('target_profit_pct', 0.01)
    stop_loss_pct = params.get('stop_loss_pct', 0.01)
    max_shares_per_trade = params.get('max_shares_per_trade', 50)
    initial_capital = params.get('capital', 10000.0)
    commission_per_share = params.get('commission_per_share', 0.005)
    
    # Load model artifacts
    model_path = f'lgbm_final_model_enhanced_{ticker.lower()}.joblib'
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Skipping {ticker} on {date}.")
        return None, None, None
    
    try:
        model, scaler, features_to_use = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return None, None, None

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
    
    # Debug code to understand feature mismatch issues
    print(f"Model expects {len(features_to_use)} features: {features_to_use}")
    print(f"Current dataframe has {df_features_model.shape[1]} features: {df_features_model.columns.tolist()}")
    
    # Scale features for prediction - only use the original features for scaling
    features_for_scaling = df_features_model[features_to_use].copy()
    print(f"Features for scaling: {features_for_scaling.shape[1]} features")
    
    df_features_scaled = scaler.transform(features_for_scaling)
    
    # Make predictions
    predictions = model.predict_proba(df_features_scaled)[:, 1]  # Get probability of positive class
    df_features_model['predicted_probability'] = predictions
    
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
                if row['regime'] == 'LOW_VOL':
                    position_size_factor = 1.0  # Full size in low vol
                elif row['regime'] == 'MED_VOL':
                    position_size_factor = 0.8  # 80% size in medium vol
                else:  # HIGH_VOL
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
    
    try:
        choice = int(input("\nEnter your choice (1-7): "))
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
        
        start_time = time.time()
        trades_df, metrics_df, viz_data = simulate_single_day(ticker, date, fetch_missing=fetch_missing)
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
        # 7. Paper Trading Integration
        ticker = input("Enter ticker (default: TSLA): ") or "TSLA"
        date_str = input("Enter date (YYYY-MM-DD) (default: 2025-05-02): ") or "2025-05-02"
        date = pd.Timestamp(date_str, tz='UTC')
        
        print(f"\nRunning simulation and integrating with paper trading for {ticker} on {date.date()}...")
        
        start_time = time.time()
        trades_df, metrics_df, _ = simulate_single_day(ticker, date)
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
    
    else:
        print("Invalid choice. Please run again and select options 1-7.")
    
    # Clean up
    if conn:
        conn.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    run_enhanced_simulation()

