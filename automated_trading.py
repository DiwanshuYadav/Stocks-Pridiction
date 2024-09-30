import alpaca_trade_api as tradeapi

def execute_trade(ticker, signal):
    api = tradeapi.REST('YOUR_API_KEY', 'YOUR_SECRET_KEY', base_url='https://paper-api.alpaca.markets')
    if signal == 'buy':
        api.submit_order(
            symbol=ticker,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
    elif signal == 'sell':
        api.submit_order(
            symbol=ticker,
            qty=1,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
