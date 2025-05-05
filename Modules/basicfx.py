

# Functions
def get_stock_data_v1(ticker, start_date, end_date, interval='1d'):
    """ Get stock data using yfinance """
    data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, interval=interval)['Close']
    return data

def get_multiple_stocks_data(tickers, start_date, end_date, interval = '1d'):
    data_df = pd.DataFrame()
    for tk in tickers:
        stock_data = get_stock_data(tk, start_date, end_date, interval=interval)
        data_df[tk] = stock_data
    return data_df

def get_stock_data(ticker, start_date, end_date, interval='1d'):
    """Download Adjusted Close prices for a single ticker."""
    stock = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if stock.empty:
        return None
    return stock['Close']

def get_log_returns(tickers, start_date, end_date, interval='1d'):
    """Get log returns for multiple tickers, ignoring download errors."""
    # Diccionario para almacenar precios
    data = {}
    failed_tickers = []

    for ticker in tickers:
        try:
            stock_data = get_stock_data(ticker, start_date, end_date, interval=interval)
            if stock_data is None:
                print(f"No data for {ticker}")
                failed_tickers.append(ticker)
                continue
            data[ticker] = stock_data
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            failed_tickers.append(ticker)

    # Si no descargamos nada, avisar
    if not data:
        raise ValueError("No se pudo descargar datos de ningún ticker.")

    # Juntar todo en un solo DataFrame
    df_prices = pd.concat(data, axis=1)

    # Calcular log returns
    log_returns = np.log(1 + df_prices.pct_change()).dropna()

    return log_returns  # Devuelve un DataFrame

def get_betas_def(stocks, benchmark, log_returns):
    """ Example structure for get_betas """
    
    betas_dict = {}
    for col in stocks:
        cov = log_returns[[col, benchmark]].cov()
        mkt_var = cov.loc[benchmark, benchmark]
        beta = cov.loc[col, benchmark] / mkt_var
        betas_dict[col] = beta
    
    betas = pd.DataFrame(betas_dict, index=['Beta'])
    return betas

def covariance_matrix(log_returns):
    """ Calculate the annualized covariance matrix of log returns """
    # Calculate the covariance matrix of log returns
    cov_matrix = log_returns.cov()
    bench_cov_matrix_excluded = cov_matrix.iloc[1:, 1:]
    return cov_matrix, bench_cov_matrix_excluded

def get_rf(interval = 'DGS1', key = None, observation_date=None, end_date = None, frequency = 1): # Default is 1 year
    """ Interval: DGS1MO (1 month), TB3MS (3 month), TB6MS (6 month), DGS1 (1 year)
    --- 
    Depending on the frequency that you put you will get either annual or daily"""
    
    fred = Fred(api_key=key)
    rf_series = fred.get_series(interval, observation_start=observation_date, observation_end=end_date) / 100
    return (1+rf_series.iloc[-1])**(1/frequency) -1 # Return the last value

def max_sharpe_ratio(port_returns, rf_annual, port_vols, weight_matrix):
    """
    Finds the index of the portfolio with the maximum Sharpe ratio.
    
    Parameters
    ----------
    port_returns : array-like
        Portfolio returns (in %).
    rf_annual : float
        Annual risk-free rate (in %).
    port_vols : array-like
        Portfolio volatilities (in %).

    Returns
    -------
    int
        Index of the portfolio with the maximum Sharpe ratio.
    """

    sharpe_ratios = (port_returns - rf_annual) / port_vols
    # Find the index of the maximum Sharpe ratio
    # Identify the maximum Sharpe ratio portfolio
    max_sr_index = np.argmax(sharpe_ratios)
    
    max_sr_vol = port_vols[max_sr_index]
    max_sr_ret = port_returns[max_sr_index]
    max_sr_weights = weight_matrix[max_sr_index]
    max_sr = sharpe_ratios[max_sr_index]

    return max_sr_ret, max_sr_vol, max_sr_weights, max_sr, sharpe_ratios
