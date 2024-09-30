def risk_management(returns, stop_loss=0.1, take_profit=0.2):
    risk_metrics = {}
    risk_metrics['Max Drawdown'] = (returns.min() - returns.max()) / returns.max()
    risk_metrics['Sharpe Ratio'] = returns.mean() / returns.std()
    return risk_metrics
