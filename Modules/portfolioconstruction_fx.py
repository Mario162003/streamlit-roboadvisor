
from Modules.imports import *
from Modules.portfolioconstruction_fx import *

def get_betas_def_v3(etf_benchmark_df, etf_log_returns, benchmark_log_returns):
    """
    Calcula la beta de cada ETF respecto a su benchmark individual, 
    alineando las fechas correctamente.
    """
    betas = {}

    for idx, row in etf_benchmark_df.iterrows():
        etf = row['Ticker']
        benchmark = row['Benchmark Ticker']

        if etf in etf_log_returns.columns and benchmark in benchmark_log_returns.columns:
            # Alinear fechas
            etf_ret = etf_log_returns[etf].dropna()
            bench_ret = benchmark_log_returns[benchmark].dropna()
            etf_ret, bench_ret = etf_ret.align(bench_ret, join='inner')

            if len(etf_ret) < 2:  # Seguridad: necesitamos al menos 2 puntos
                print(f"⚠️ Warning: Not enough data for {etf} and {benchmark}")
                continue

            cov = np.cov(etf_ret, bench_ret)
            beta = cov[0, 1] / cov[1, 1]
            betas[etf] = beta
        else:
            print(f"⚠️ Warning: {etf} or {benchmark} not found in returns data.")

    betas_df = pd.DataFrame(betas, index=['Beta'])

    return betas_df

def get_stock_estimates_df_v3(etf_benchmark_df, etf_log_returns, benchmark_log_returns, rf, betas, frequency=252):
    """
    Calcula Expected Return, Sharpe Ratio y Volatility de cada ETF, 
    usando su benchmark y alineando fechas correctamente.
    """
    results = []

    for idx, row in etf_benchmark_df.iterrows():
        etf = row['Ticker']
        benchmark = row['Benchmark Ticker']

        if etf in etf_log_returns.columns and benchmark in benchmark_log_returns.columns:
            returns = etf_log_returns[etf].dropna()
            benchmark_returns = benchmark_log_returns[benchmark].dropna()

            # Alinear fechas
            returns, benchmark_returns = returns.align(benchmark_returns, join='inner')

            if len(returns) < 2:
                print(f"⚠️ Warning: Not enough data for {etf} and {benchmark}")
                continue

            beta = betas.loc['Beta', etf]

            expected_return = rf + beta * (benchmark_returns.mean() - rf)
            expected_annual_return = (1 + expected_return)**frequency - 1

            annual_rf = (1 + rf)**frequency - 1
            volatility = returns.std()
            annual_volatility = volatility * np.sqrt(frequency)
            sharpe_ratio = (expected_annual_return - annual_rf) / annual_volatility if annual_volatility != 0 else np.nan

            results.append({
                'Ticker': etf,
                'Expected Return': expected_annual_return,
                'Sharpe Ratio': sharpe_ratio,
                'Volatility': annual_volatility
            })
        else:
            print(f"⚠️ Warning: {etf} or {benchmark} not found in returns data.")

    estimates_df = pd.DataFrame(results).set_index('Ticker')

    return estimates_df

def get_bond_estimates_df(
    etf_returns_df, 
    risk_free_rate=0.02, 
    periods_per_year=12
):
    """
    Calculates Expected Return, Sharpe Ratio, and Volatility for a DataFrame of Bond ETFs.
    
    Args:
        etf_returns_df (pd.DataFrame): DataFrame with ETFs as columns and returns as rows (indexed by Date).
        risk_free_rate (float): Annualized risk-free rate (default = 2%).
        periods_per_year (int): Number of periods per year (12 for monthly, 252 for daily).
    
    Returns:
        pd.DataFrame: DataFrame with ETFs as index and Expected Return, Sharpe Ratio, Volatility as columns.
    """
    
    results = []

    for ticker in etf_returns_df.columns:
        returns = etf_returns_df[ticker].dropna()
        
        if returns.empty:
            continue  # Skip ETFs with no valid data
        
        # Annualized calculations
        mean_return = returns.mean() * periods_per_year
        std_dev = returns.std() * np.sqrt(periods_per_year)
        
        excess_return = mean_return - risk_free_rate
        sharpe_ratio = excess_return / std_dev if std_dev != 0 else np.nan
        
        results.append({
            'Ticker': ticker,
            'Expected Return': mean_return,
            'Sharpe Ratio': sharpe_ratio,
            'Volatility': std_dev
        })
    
    results_df = pd.DataFrame(results)
    results_df.set_index('Ticker', inplace=True)
    
    return results_df

def get_stock_estimates_df(tickers, rf, benchmark_ticker, log_returns, betas, frequency=1):
    """ Calculate expected return, Sharpe ratio, and volatility for each stock """
    results = []

    for stock in tickers:
        expected_return = rf + betas.loc['Beta', stock] * (log_returns[benchmark_ticker].mean() - rf)
        expected_annual_return = (1 + expected_return) ** frequency - 1
        
        annual_rf = (1 + rf) ** frequency - 1
        annual_sharpe_ratio = (expected_annual_return - annual_rf) / (log_returns[stock].std() * np.sqrt(frequency))
        
        volatility = log_returns[stock].std()
        annual_vol = volatility * np.sqrt(frequency)
        
        results.append({
            'Ticker': stock,
            'Expected Return': expected_annual_return,
            'Sharpe Ratio': annual_sharpe_ratio,
            'Volatility': annual_vol
        })
    
    estimates = pd.DataFrame(results).set_index('Ticker')
    return estimates

def select_diversified_etfs_v3(
    ranked_etfs, 
    returns_df, 
    asset_class='equity', 
    n_etfs=10, 
    base_correlation_threshold=0.8, 
    volatility_penalty_factor=0.2,
    verbose=True
):
    """
    Selecciona ETFs diversificados maximizando Score ajustado por volatilidad y reduciendo correlación redundante.

    Args:
        ranked_etfs (pd.DataFrame): DataFrame con columnas ['Score', 'Volatility'] mínimo.
        returns_df (pd.DataFrame): DataFrame de log-returns (ETFs como columnas).
        asset_class (str): 'equity' o 'bond'.
        n_etfs (int): Número de ETFs a seleccionar.
        base_correlation_threshold (float): Umbral de correlación base permitido.
        volatility_penalty_factor (float): Factor de penalización de volatilidad en el Score.
        verbose (bool): Mostrar progreso.

    Returns:
        list: Lista de ETFs seleccionados.
    """

    # Ajustar threshold para bonds más estricto
    if asset_class.lower() == 'equity':
        correlation_threshold = base_correlation_threshold
    elif asset_class.lower() == 'bond':
        correlation_threshold = base_correlation_threshold - 0.2
    else:
        raise ValueError("asset_class must be 'equity' or 'bond'")

    # Clonar y ajustar Score penalizando Volatility
    ranked_etfs = ranked_etfs.copy()
    if 'Volatility' in ranked_etfs.columns:
        ranked_etfs['Volatility Norm'] = (
            (ranked_etfs['Volatility'] - ranked_etfs['Volatility'].min()) / 
            (ranked_etfs['Volatility'].max() - ranked_etfs['Volatility'].min())
        )
        ranked_etfs['Adjusted Score'] = ranked_etfs['Score'] - volatility_penalty_factor * ranked_etfs['Volatility Norm']
    else:
        ranked_etfs['Adjusted Score'] = ranked_etfs['Score']

    # Reordenar
    ranked_etfs = ranked_etfs.sort_values(by='Adjusted Score', ascending=False)

    selected = []

    # Selección principal
    for etf in ranked_etfs.index:
        if len(selected) == 0:
            selected.append(etf)
            if verbose:
                print(f"✅ Adding {etf} (first ETF)")
            continue
        
        correlations = returns_df[selected].corrwith(returns_df[etf])

        if correlations.abs().max() < correlation_threshold:
            selected.append(etf)
            if verbose:
                print(f"✅ Adding {etf} (max corr={correlations.abs().max():.2f})")
        else:
            if verbose:
                print(f"🚫 Skipping {etf} (max corr={correlations.abs().max():.2f})")

        if len(selected) >= n_etfs:
            break

    if verbose:
        print(f"\n🎯 Selected {len(selected)} ETFs out of {len(ranked_etfs)} candidates.")

    return selected

def select_diversified_etfs(
    ranked_etfs, 
    returns_df, 
    asset_class='equity', 
    n_etfs=5, 
    base_correlation_threshold=0.8, 
    volatility_penalty_factor=0.2
):
    """
    Selects a diversified set of ETFs with high scores and low correlation,
    adjusting for volatility.

    Args:
        ranked_etfs (pd.DataFrame): DataFrame with ETFs ranked by Score ('Score' column).
        returns_df (pd.DataFrame): DataFrame of ETF returns (columns = tickers).
        asset_class (str): 'equity' or 'bond'.
        n_etfs (int): Number of ETFs to select.
        base_correlation_threshold (float): Base max correlation allowed.
        volatility_penalty_factor (float): Penalty to apply to volatility in Score adjustment.

    Returns:
        list: Selected ETF tickers.
    """
    # Adjust correlation threshold dynamically
    if asset_class.lower() == 'equity':
        correlation_threshold = base_correlation_threshold  # e.g., 0.8
    elif asset_class.lower() == 'bond':
        correlation_threshold = base_correlation_threshold - 0.2  # e.g., 0.6
    else:
        raise ValueError("asset_class must be 'equity' or 'bond'")

    selected = []
    
    # Adjust the Score penalizing volatility
    ranked_etfs = ranked_etfs.copy()
    if 'Volatility' in ranked_etfs.columns:
        # Normalize Volatility
        ranked_etfs['Volatility Norm'] = (ranked_etfs['Volatility'] - ranked_etfs['Volatility'].min()) / (ranked_etfs['Volatility'].max() - ranked_etfs['Volatility'].min())
        # Penalize Score
        ranked_etfs['Adjusted Score'] = ranked_etfs['Score'] - volatility_penalty_factor * ranked_etfs['Volatility Norm']
        # Reorder ETFs by Adjusted Score
        ranked_etfs = ranked_etfs.sort_values(by='Adjusted Score', ascending=False)
    else:
        ranked_etfs['Adjusted Score'] = ranked_etfs['Score']

    # Selection loop
    for etf in ranked_etfs.index:
        if len(selected) == 0:
            selected.append(etf)
        else:
            correlations = returns_df[selected].corrwith(returns_df[etf])
            if all(abs(correlations) < correlation_threshold):
                selected.append(etf)
        
        if len(selected) >= n_etfs:
            break

    return selected

def select_low_correlation_etfs(
    ranked_etfs, 
    returns_df, 
    n_etfs=5, 
    correlation_threshold=0.8
):
    """
    Selects a set of ETFs with high score but low correlation between each other.
    
    Args:
        ranked_etfs (pd.DataFrame): Ranked DataFrame with 'Score' and index = ETF tickers.
        returns_df (pd.DataFrame): DataFrame of returns (columns = ETFs).
        n_etfs (int): Number of ETFs to select.
        correlation_threshold (float): Maximum allowed correlation between selected ETFs.
    
    Returns:
        list: Selected ETF tickers.
    """
    selected = []
    
    for etf in ranked_etfs.index:
        if len(selected) == 0:
            selected.append(etf)
        else:
            correlations = returns_df[selected].corrwith(returns_df[etf])
            if all(abs(correlations) < correlation_threshold):
                selected.append(etf)
        
        if len(selected) >= n_etfs:
            break
    
    return selected

def optimize_profiled_portfolio_3(
    log_returns, expected_returns, cov_matrix, rf,
    bonds, stocks, profile_name, profile_allocation,
    min_weight=0.0
):
    """
    Optimiza una cartera según perfil de riesgo con restricciones de asignación FI/EQ
    y peso mínimo por activo si se desea.
    """

    all_assets = bonds + stocks

    # Verificación de activos duplicados
    assert len(set(all_assets)) == len(all_assets), f"❌ Activos duplicados en {profile_name}"

    mu = expected_returns.loc[all_assets, 'Expected Return'].values / 100  # en decimal
    cov = cov_matrix.loc[all_assets, all_assets]
    n = len(all_assets)

    # Definir bounds con min_weight
    bounds = [(min_weight, 1) for _ in range(n)]

    # Índices para restricciones
    bond_idx = [i for i, a in enumerate(all_assets) if a in bonds]
    stock_idx = [i for i, a in enumerate(all_assets) if a in stocks]

    def neg_sharpe(w):
        try:
            port_ret = np.dot(w, mu).item()
            port_vol = np.sqrt(w @ cov @ w).item()
            if port_vol == 0:
                return np.inf
            return -((port_ret - rf) / port_vol)
        except Exception as e:
            print(f"⚠️ Error en función objetivo: {e}")
            return np.inf

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w[bond_idx]) - profile_allocation["FI"]},
        {"type": "eq", "fun": lambda w: np.sum(w[stock_idx]) - profile_allocation["EQ"]},
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    init_guess = np.ones(n) / n
    result = sco.minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError(f"❌ Optimization failed for {profile_name}: {result.message}")

    optimal_w = result.x
    port_ret = np.dot(optimal_w, mu).item()
    port_vol = np.sqrt(optimal_w @ cov @ optimal_w).item()
    port_sr = (port_ret - rf) / port_vol

    weights_series = pd.Series(optimal_w, index=all_assets)

    # 🧠 Mini log de control:
    invested_assets = weights_series[weights_series > 1e-6]
    n_invested = len(invested_assets)
    max_weight_asset = invested_assets.idxmax()
    min_weight_asset = invested_assets.idxmin()

    print(f"🔎 {profile_name}: {n_invested} assets invested.")
    print(f"   ➔ Max Weight: {max_weight_asset} ({invested_assets[max_weight_asset]*100:.2f}%)")
    print(f"   ➔ Min Weight: {min_weight_asset} ({invested_assets[min_weight_asset]*100:.2f}%)")

    return {
        "profile": profile_name,
        "weights": weights_series,
        "return": port_ret * 100,
        "volatility": port_vol * 100,
        "sharpe": port_sr
    }

























































