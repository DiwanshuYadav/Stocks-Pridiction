import argparse
from data_loader import fetch_stock_data, preprocess_data
from feature_engineering import calculate_technical_indicators
from modeling import prepare_lstm_data, build_lstm_model
from backtesting import backtest_strategy
from predictor import plot_predictions, evaluate_model
from news_sentiment import fetch_news_sentiment
from portfolio_management import optimize_portfolio
from risk_management import risk_management
from automated_trading import execute_trade
from visualization import create_dashboard, plot_stock_chart, plot_candlestick_chart, plot_sentiment

def main(ticker, period='1y', data_type='stock'):
    # Data Preparation
    print(f"Fetching data for {ticker} with period {period}...")
    stock_data = fetch_stock_data(ticker, period, data_type)
    stock_data = preprocess_data(stock_data)
    stock_data = calculate_technical_indicators(stock_data)
    
    # News Sentiment Analysis
    sentiment = fetch_news_sentiment(ticker)
    print(f"Average sentiment for {ticker}: {sentiment}")
    
    # Model Preparation and Training
    X, y, scaler = prepare_lstm_data(stock_data)
    model = build_lstm_model(X.shape)
    model.fit(X, y, epochs=50, batch_size=32)
    
    # Predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    # Evaluation
    evaluate_model(y, predictions)
    
    # Backtesting
    backtest_strategy(stock_data)
    
    # Visualization
    plot_predictions(stock_data, predictions, scaler)
    
    # Portfolio Management
    expected_return, risk = optimize_portfolio(predictions, stock_data)
    print(f"Expected Return: {expected_return}, Risk: {risk}")
    
    # Risk Management
    risk_metrics = risk_management(pd.Series(predictions))
    print(f"Risk Metrics: {risk_metrics}")
    
    # Automated Trading (Example)
    signal = 'buy' if expected_return > 0 else 'sell'
    execute_trade(ticker, signal)
    
    # Enhanced Visualization and Dashboard
    create_dashboard(stock_data, pd.Series([sentiment] * len(stock_data), index=stock_data.index))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Market Predictor')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='1y', help='Data period (e.g., 1y, 6mo)')
    parser.add_argument('--type', type=str, choices=['stock', 'futures', 'options'], default='stock', help='Type of data to fetch')
    
    args = parser.parse_args()
    
    main(args.ticker, args.period, args.type)
