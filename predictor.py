import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_predictions(stock_data, predictions, scaler):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index, stock_data['Close'], label='Actual Prices')
    plt.plot(stock_data.index[-len(predictions):], scaler.inverse_transform(predictions), label='Predicted Prices')
    plt.legend()
    plt.show()

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"RMSE: {rmse}, MAE: {mae}")
