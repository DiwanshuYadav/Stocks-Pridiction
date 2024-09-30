import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Function to create an interactive stock price chart
def plot_stock_chart(stock_data):
    fig = go.Figure()

    # Plotting the closing price
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))

    # Plotting technical indicators
    if 'MA20' in stock_data.columns:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA20'], mode='lines', name='MA20'))
    if 'MA50' in stock_data.columns:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='MA50'))

    fig.update_layout(title='Stock Price and Technical Indicators', xaxis_title='Date', yaxis_title='Price')
    return fig

# Function to create a candlestick chart
def plot_candlestick_chart(stock_data):
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    )])

    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

# Function to create a sentiment analysis chart
def plot_sentiment(sentiments):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sentiments.index, sentiments.values, marker='o', linestyle='-', color='b')
    ax.set_title('Sentiment Analysis Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Score')

    # Convert matplotlib figure to PNG image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return f"data:image/png;base64,{img_str}"

# Function to create a dashboard using Dash
def create_dashboard(stock_data, sentiments):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Stock Market Dashboard"),
        dcc.Graph(figure=plot_stock_chart(stock_data), id='stock-chart'),
        dcc.Graph(figure=plot_candlestick_chart(stock_data), id='candlestick-chart'),
        html.Img(src=plot_sentiment(sentiments), style={'width': '100%', 'height': 'auto'}),
    ])

    app.run_server(debug=True)

# Example usage
if __name__ == "__main__":
    # Sample data for demonstration
    dates = pd.date_range(start='2023-01-01', periods=100)
    stock_data = pd.DataFrame(index=dates, data={
        'Open': np.random.randn(100).cumsum(),
        'High': np.random.randn(100).cumsum(),
        'Low': np.random.randn(100).cumsum(),
        'Close': np.random.randn(100).cumsum()
    })
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()

    sentiments = pd.Series(np.random.randn(100), index=dates)

    create_dashboard(stock_data, sentiments)
