
## Prerequisites

*   **Git:** For cloning the repository.
*   **Docker & Docker Compose:** For running the database and Java service. (Install from [docker.com](https://www.docker.com/))
*   **Java Development Kit (JDK):** Version 21 or higher (required by the Java service).
*   **Python:** Version 3.10 or higher.
*   **Python Virtual Environment:** Recommended (`python3 -m venv venv`).
*   **pip:** Python package installer.
*   **Polygon.io Account:** A **paid** subscription plan that allows access to real-time WebSocket streams (e.g., "Stocks Starter" or higher). You will need your API Key.
*   **(Optional) Compute Resources:** Training ML models, especially on large datasets, can be resource-intensive. A GPU is recommended for faster training if using GPU-enabled libraries (like TensorFlow, or correctly compiled XGBoost/LightGBM).

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory> 
    ```

2.  **Configuration:**
    *   **API Key:** Create a file named `.env` in the project root directory. Add your Polygon.io API key:
        ```dotenv
        POLYGON_API_KEY=YOUR_POLYGON_API_KEY_HERE
        ```
        **(Security:** Add `.env` to your `.gitignore` file immediately!)
    *   **Database Password (Optional but recommended):** Add a secure password for the database to the `.env` file (the Python scripts will use this):
        ```dotenv
        POLYGON_DB_PASSWORD=your_super_secret_db_password 
        ```
    *   **Docker Compose (`docker-compose.yml`):**
        *   Verify the `environment` section for the `ticker-service`. Ensure `POLYGON_API_KEY` is set correctly (it should read from `.env` automatically if configured, otherwise set it directly here - **less secure**).
        *   Update `POLYGON_TICKERS` to the list of symbols you want the *Java service* to collect aggregates for (e.g., `TSLA,AAPL`).
        *   Update the `environment` section for the `timescaledb` service. Change `POSTGRES_PASSWORD` to match the secure password you set in `.env`. Ensure `POSTGRES_USER` and `POSTGRES_DB` match what the Java service and Python scripts expect (`polygonuser`, `polygondata` in our examples).
        *   Check the mapped host port for `timescaledb` (e.g., `5433:5432`) and `grafana` (`3000:3000`).

3.  **Build Java Service Docker Image:** Navigate to the Java service directory (assuming it's the root of *this* repo, otherwise adjust `cd` path) and build the image using Gradle:
    ```bash
    # cd Polygon-Ticker-Service # If Java service is in a subfolder
    chmod +x ./gradlew
    ./gradlew clean bootBuildImage 
    ```
    *(This compiles the Java code we added and creates a Docker image named `polygon-ticker-service:0.0.1-SNAPSHOT`)*

4.  **Start Infrastructure Services:** Run Docker Compose to start the TimescaleDB database and the Java data collection service:
    ```bash
    docker-compose up -d
    ```
    *   Check logs to ensure they start correctly: `docker-compose logs -f ticker-service` and `docker-compose logs -f timescaledb`. Look for successful DB initialization and Java service startup.

## Training Your Model

**This step is MANDATORY.** Due to size, data specificity, and market changes, a pre-trained model is not included. You must train your own model using historical data relevant to your desired prediction period.

1.  **Ensure Data Exists:** Let the Java service run during market hours to collect sufficient 1-minute aggregate data (`stock_aggregates_min` table) **OR** load historical data manually (like we did using `fetch_polygon_data.py` and `load_aggregates.py` - you might need to create/run similar scripts). You need several weeks/months of data for good results.

2.  **Set Up Python Environment:**
    ```bash
    # Navigate to where your Python scripts are (e.g., project root or 'sonnet' subfolder)
    # cd sonnet 
    python3 -m venv venv       # Create virtual environment
    source venv/bin/activate   # Activate it
    # Install all dependencies
    pip install lightgbm psycopg2-binary python-dotenv pandas numpy scikit-learn matplotlib joblib pandas-ta 
    ```

3.  **Run the Training Script:** Execute the script that performs walk-forward validation and saves the final model artifacts (using LightGBM in our last example):
    ```bash
    python3 lightgbm_alternative_targets.py # Or xgboost_walk_forward.py, etc.
    ```
    *   **Choose `TARGET_TYPE`** inside the script (`binary`, `multiclass`, `regression`) before running.
    *   This script will load data, create features, run walk-forward validation, print results, potentially train a final model, and crucially, save the trained model, scaler, and feature list (e.g., to `lgbm_multiclass_model_tsla.joblib`). **Make note of this filename.**

## Running the Prediction Service

This service uses the trained model to make predictions on new data coming into the database.

1.  **Ensure Infrastructure is Running:** `docker-compose up -d` (DB and Java Service).
2.  **Verify Artifacts:** Make sure the correct model artifact file (e.g., `lgbm_multiclass_model_tsla.joblib`) exists and its path is correctly set in `MODEL_ARTIFACT_PATH` inside the prediction script.
3.  **Activate Python Environment:**
    ```bash
    # Navigate to script directory
    # cd sonnet
    source ../venv/bin/activate # Activate venv from project root
    ```
4.  **Run the Prediction Script:**
    ```bash
    python3 lgbm_prediction_service.py 
    ```
    *   The script will connect to the DB, load the model/scaler, and enter a loop.
    *   Every `PREDICTION_INTERVAL_SECONDS` (default 60s), it fetches the latest data, generates features, predicts, and saves the result to the `stock_predictions` table.
    *   The main thread will display the latest few predictions from the database. Press `Ctrl+C` to stop.

## Running the Paper Trader (Optional)

This simulates trading based on the signals generated by the prediction service.

1.  **Ensure Infrastructure & Prediction Service Are Running:** Both the Docker services and `lgbm_prediction_service.py` must be running.
2.  **Open a *New* Terminal.**
3.  **Activate Python Environment:**
    ```bash
    cd <your-repo-directory> # e.g., ~/Polygon-Ticker-Service
    source venv/bin/activate
    # cd sonnet # If script is in subfolder
    ```
4.  **Run the Paper Trading Script:**
    ```bash
    python3 paper_trading_test.py 
    ```
    *   The script will connect to the DB and create the `paper_trades` table.
    *   It enters a loop, checking the `stock_predictions` table for new BUY signals every ~30 seconds.
    *   It also checks open positions against latest prices to simulate closing trades on target/stop-loss/timeout.
    *   It periodically prints portfolio status. Press `Ctrl+C` to stop.

## Visualization (Grafana)

1.  **Access Grafana:** Open `http://localhost:3000` in your browser.
2.  **Add Data Source:**
    *   Go to Configuration (Cog icon) -> Data Sources -> Add data source.
    *   Select "PostgreSQL".
    *   **Host:** `timescaledb:5432` (Use the Docker service name and *internal* port).
    *   **Database:** `polygondata`
    *   **User:** `polygonuser`
    *   **Password:** The password you set for `POSTGRES_PASSWORD` (or `.env`).
    *   **TLS/SSL Mode:** `disable`
    *   **TimescaleDB:** Toggle **ON**.
    *   Click "Save & test".
3.  **Create Dashboard & Panels:**
    *   Create a new dashboard.
    *   **Panel 1: Price/Candlestick (from Aggregates):**
        *   Visualization: Candlestick
        *   Query:
          ```sql
          SELECT
            start_time AS "time",
            agg_open AS "Open",
            agg_high AS "High",
            agg_low AS "Low",
            agg_close AS "Close"
          FROM stock_aggregates_min
          WHERE symbol = 'TSLA' AND $__timeFilter(start_time)
          ORDER BY time ASC;
          ```
    *   **Panel 2: Prediction Probability/Signal:**
        *   Visualization: Time series
        *   Query:
          ```sql
          SELECT 
              timestamp as time,
              predicted_probability, 
              CASE WHEN trade_signal THEN 1 ELSE 0 END AS signal -- Map boolean to 0/1 for plotting
          FROM stock_predictions
          WHERE ticker = 'TSLA' AND $__timeFilter(timestamp)
          ORDER BY time ASC;
          ```
        *   Configure Y-axes appropriately (e.g., left for probability 0-1, right for signal 0/1). Use overrides to style the 'signal' series as points or steps.
    *   **Panel 3: Paper Trade P&L (Example):**
        *   Visualization: Stat or Table
        *   Query:
          ```sql
          SELECT SUM(profit_loss) as total_pnl 
          FROM paper_trades 
          WHERE trade_status != 'OPEN' AND $__timeFilter(exit_time);
          ```
    *   Set dashboard time range and refresh rate.

## Important Considerations

*   **NOT FINANCIAL ADVICE:** This is an example project for technical demonstration and learning. It is **NOT** financial advice. Trading involves significant risk, and automated systems can lose money rapidly. Do not use this code for live trading without extensive testing, validation, and understanding the risks.
*   **Data Costs:** Real-time data from Polygon.io requires a paid subscription. Fetching large amounts of historical data via the REST API can also consume API credits quickly.
*   **Simulation != Reality:** Backtesting and paper trading results do not guarantee live performance. They do not account for slippage, commissions, data feed latency, API errors, or the psychological aspects of live trading.
*   **Model Drift:** Market dynamics change. Models trained on historical data will likely degrade over time ("drift") and require periodic retraining with fresh data.
*   **API Limits:** Be mindful of Polygon.io's API rate limits. Add delays (`time.sleep`) in loops if necessary.
*   **Error Handling:** The provided scripts have basic error handling. Production systems require much more robust error detection, logging, and recovery mechanisms.
*   **Security:** Protect your Polygon.io API key. Do not commit it directly to Git. Use environment variables or secure secret management.

## Future Improvements / TODO

*   Implement hyperparameter tuning for LightGBM/XGBoost using walk-forward cross-validation (e.g., `TimeSeriesSplit` with `GridSearchCV`).
*   Experiment with different target variables (regression, direction prediction).
*   Try other models (e.g., TabPFN, Neural Networks with Attention).
*   Implement regime-specific models or weighting.
*   Use a dedicated backtesting library (Backtrader, VectorBT) for more realistic simulations including costs and portfolio management.
*   Enhance the Java service's robustness and add configuration options.
*   Improve error handling and logging throughout the Python scripts.
*   Develop a more sophisticated position sizing strategy for the paper trader.
*   Build a proper web UI frontend instead of relying solely on Grafana/terminal output.

## License
MIT 
