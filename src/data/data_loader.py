import yfinance as yf

if __name__=='__main__':
    data = yf.download('BBCA', start='2020-01-01', end='2025-07-25')
    data.to_csv('data/BBCA_historical_data.csv')