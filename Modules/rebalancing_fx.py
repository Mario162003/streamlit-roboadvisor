

from Modules.imports import *
from Modules.basicfx import *
from Modules.portfolioconstruction_fx import *

def get_estimates_rebalance(tickers, rf, benchmark_ticker, log_returns, betas, frequency=1):
    """ Calculate expected return, Sharpe ratio, and volatility for each stock """
    # Calculate the expected return, Sharpe ratio, and volatility for each stock
    # Create an empty DataFrame to store estimates
    estimates = pd.DataFrame(columns=['Expected Return', 'Sharpe Ratio', 'Volatility'])

    # Loop through each stock (excluding the market index)
    for stock in tickers[1:]:
    # CAPM Expected Return: rf + beta * (market_return - rf)
        expected_return = rf + betas.loc['Beta', stock] * (log_returns[benchmark_ticker].mean() - rf)
        expected_annual_return = (1 + expected_return) ** frequency - 1
        # Sharpe Ratio: (expected return - rf) / volatility
        #sharpe_ratio = (expected_return - rf) / (log_returns[stock].std())
        annual_rf = (1 + rf) ** frequency - 1
        # Annualized Sharpe Ratio
        annual_sharpe_ratio = (expected_annual_return - annual_rf) / (log_returns[stock].std() * np.sqrt(frequency))
        # Volatility: std of log returns
        volatility = log_returns[stock].std()
        annual_vol = volatility * np.sqrt(frequency)
        # Add the row to the DataFrame
        estimates.loc[stock] = [expected_annual_return, annual_sharpe_ratio, annual_vol]

def calc_betas(log_returns: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """
    Returns a 1‑row DataFrame: index='Beta', columns = tickers (incl. benchmark).
    beta(benchmark)=1 by definition.
    """
    cov = log_returns.cov()
    betas = pd.DataFrame(index=['Beta'])
    mkt_var = cov.loc[benchmark, benchmark]
    for col in log_returns.columns:
        betas[col] = cov.loc[col, benchmark] / mkt_var
    betas[benchmark] = 1.0
    return betas

# ---------- helpers ------------------------------------------------------
def build_cash_schedule(index, cash_amount, cash_freq="BME", first_cash_date=None):
    if cash_amount == 0:
        return pd.Series(0.0, index=index)
    inj = index.to_series().resample(cash_freq).last().index
    if first_cash_date is not None:
        inj = inj[inj >= pd.to_datetime(first_cash_date)]
    s = pd.Series(0.0, index=index)
    s.loc[inj.intersection(index)] = cash_amount
    return s

def worst_sharpe_asset(log_returns_30d):
    """Devuelve el ticker con menor Sharpe (µ/σ) en la ventana de 30 días."""
    sr = (log_returns_30d.mean() / log_returns_30d.std()).replace([np.inf, -np.inf], np.nan)
    return sr.idxmin()

def bck_test_final_DYNAMIC_v2(
    PV0: float,
    price_data: pd.DataFrame,
    rf_annual: float,
    initial_weights: pd.Series | dict,
    FI_TICKERS: list[str],
    EQ_TICKERS: list[str],
    risk_profile: str,
    PROFILE_TARGET: dict,
    lookback_days: int = 252,
    min_obs: int = 60,
    opt_freq: str = "BYE",
    rebalance_freq: str = "BME",
    cash_amount: float = 0.0,
    cash_freq: str = "BME",
    first_cash_date=None,
    cost_rate: float = 0.001,
    benchmark: str = "^GSPC",
    min_weight: float = 0.035,
    fallback_equal_weight: bool = True
):
    """- 'BME' : Business Month End (fin de mes laboral)
        - 'BMS' : Business Month Start (inicio de mes laboral)
        - 'BYE' : Business Year End (fin de año laboral)
        - 'BYS' : Business Year Start (inicio de año laboral)
        - 'BQE' or 'BQ' : Business Quarter End (fin de trimestre laboral)"""
    
    import numpy as np
    import pandas as pd
    import scipy.optimize as sco

    # ------------------ Normalizar pesos iniciales ------------------
    tgt = {k: v for k, v in initial_weights.items() if k in price_data.columns}
    tgt = {k: w / sum(tgt.values()) for k, w in tgt.items()}
    invest = list(tgt.keys())

    all_t = [benchmark] + [t for t in invest if t != benchmark]

    opt_set = set(price_data.resample(opt_freq).last().index)
    reb_set = set(price_data.resample(rebalance_freq).last().index)

    cash_plan = build_cash_schedule(price_data.index, cash_amount, cash_freq, first_cash_date)
    cash_balance = 0.0

    first_day = price_data.index[0]
    px0 = price_data.loc[first_day]
    hold = {a: int(np.floor(tgt[a] * PV0 / px0[a])) for a in invest}
    cash_balance += PV0 - sum(hold[a] * px0[a] for a in invest)

    freq, rf_day = 252, (1 + rf_annual)**(1/252) - 1
    rows = []

    fi_mask = np.array([t in FI_TICKERS for t in invest])
    fi_target = PROFILE_TARGET[risk_profile]["FI"]

    for i, day in enumerate(price_data.index):
        cash_balance += cash_plan.loc[day]
        px = price_data.loc[day]
        pv_pos = sum(hold[a] * px[a] for a in invest)
        pv = pv_pos + cash_balance

        if day in opt_set and i > 0:
            win = np.log(price_data.iloc[max(0,i-lookback_days):i+1].pct_change() + 1).dropna()
            if len(win) >= min_obs:
                # Stocks vs Bonds
                win_stocks = win[[a for a in invest if a in EQ_TICKERS]]
                win_bonds = win[[a for a in invest if a in FI_TICKERS]]

                estimates_stocks = get_estimates_rebalance(
                    [benchmark] + win_stocks.columns.tolist(),
                    rf_day, benchmark, win[[benchmark] + win_stocks.columns.tolist()],
                    calc_betas(win[[benchmark] + win_stocks.columns.tolist()], benchmark),
                    frequency=freq
                )

                estimates_bonds = get_bond_estimates_df(win_bonds, risk_free_rate=rf_annual, periods_per_year=freq)

                # Combinar
                all_estimates = pd.concat([estimates_stocks, estimates_bonds], axis=0)

                exp_r = all_estimates.loc[invest, "Expected Return"].values / 100
                cov_m = win[invest].cov() * freq

                try:
                    opt_w = optimize_profiled_portfolio_in_backtest(
                        exp_r, cov_m, rf_annual, fi_mask, fi_target, min_weight
                    )
                    tgt = dict(zip(invest, opt_w))
                except Exception as e:
                    print(f"⚠️ Optimization failed on {day}: {e}")
                    if fallback_equal_weight:
                        # Fallback Equal Weight respecting FI/EQ
                        n_fi = sum(fi_mask)
                        n_eq = len(fi_mask) - n_fi
                        if n_fi > 0:
                            w_fi = np.ones(n_fi) / n_fi * fi_target
                        else:
                            w_fi = np.array([])
                        if n_eq > 0:
                            w_eq = np.ones(n_eq) / n_eq * (1 - fi_target)
                        else:
                            w_eq = np.array([])
                        w_total = np.concatenate([w_fi, w_eq])
                        tgt = dict(zip(invest, w_total))

        # --- Rebalanceo mensual (whole-share + coste transacción) ---
        tr_sh = {a: 0 for a in invest}
        if day in reb_set:
            cur_val = {a: hold[a]*px[a] for a in invest}
            pv = sum(cur_val.values()) + cash_balance
            des_val = {a: tgt[a]*pv for a in invest}
            cash_tr = {a: des_val[a]-cur_val[a] for a in invest}

            for a in invest:
                tr_sh[a] = int(np.floor(cash_tr[a]/px[a])) if cash_tr[a]>0 else int(np.ceil(cash_tr[a]/px[a]))

            dollars = {a: tr_sh[a]*px[a] for a in invest}
            need = sum(d for d in dollars.values() if d>0)
            recv = -sum(d for d in dollars.values() if d<0)
            gross = sum(abs(d) for d in dollars.values())
            fee = gross * cost_rate

            cash_balance += recv
            need_tot = need + fee

            if need_tot > cash_balance:
                scale = cash_balance / need_tot
                for a in invest:
                    if tr_sh[a] > 0:
                        tr_sh[a] = int(np.floor(tr_sh[a]*scale))
                dollars = {a: tr_sh[a]*px[a] for a in invest}
                need = sum(d for d in dollars.values() if d>0)
                gross = sum(abs(d) for d in dollars.values())
                fee = gross * cost_rate
                need_tot = need + fee

            cash_balance -= need_tot

            # Ventas de emergencia si cash negativo
            if cash_balance < 0:
                look30 = np.log(price_data.loc[:day].iloc[-31:].pct_change() + 1).dropna()
                worst = worst_sharpe_asset(look30[invest])
                while cash_balance < 0 and hold[worst] > 0:
                    hold[worst] -= 1
                    cash_balance += px[worst] - px[worst]*cost_rate

            for a in invest:
                hold[a] += tr_sh[a]

        # --- Log diario ---
        rec = {"Date": day,
               ("PORTFOLIO", "Value"): pv_pos + cash_balance,
               ("PORTFOLIO", "Cash"): cash_balance}
        for a in invest:
            val = hold[a] * px[a]
            rec[(a, "Weight")] = val / (pv_pos + cash_balance) if pv else 0.0
            rec[(a, "Holdings")] = hold[a]
            rec[(a, "Trade")] = tr_sh[a]

        rows.append(rec)

    df = pd.DataFrame(rows).set_index("Date")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

# Función auxiliar para la optimización dentro del backtest

def optimize_profiled_portfolio_in_backtest(exp_ret, cov, rf, fi_mask, fi_target, min_weight=0.035):
    n = len(exp_ret)

    def neg_sharpe(w):
        pr, pv = w @ exp_ret, np.sqrt(w @ cov @ w)
        return -(pr - rf) / pv

    bounds = [(min_weight, 1.0) for _ in range(n)]

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: np.sum(w[fi_mask]) - fi_target},
    ]

    res = sco.minimize(neg_sharpe, np.full(n, 1/n), bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x

def bck_test_final_STATIC_v2(
    PV0: float,
    price_data: pd.DataFrame,
    target_weights: pd.Series | dict | None = None,
    *,
    risk_profile: str | None = None,
    FI_TICKERS: list[str] | None = None,
    EQ_TICKERS: list[str] | None = None,
    rebalance_freq: str = "BME",
    cash_amount: float = 0.0,
    cash_freq: str = "BME",
    first_cash_date=None,
    cost_rate: float = 0.001,
    PROFILE_TARGET = dict
):
    import numpy as np
    import pandas as pd

    # ---------- build target weights ----------------------------------
    if target_weights is None:
        if risk_profile is None:
            raise ValueError("Must pass either target_weights or risk_profile")
        if risk_profile not in PROFILE_TARGET:
            raise ValueError(f"risk_profile must be one of {list(PROFILE_TARGET)}")

        if not FI_TICKERS or not EQ_TICKERS:
            raise ValueError("Need FI_TICKERS and EQ_TICKERS when using risk_profile")

        fi_share = PROFILE_TARGET[risk_profile]["FI"]
        eq_share = PROFILE_TARGET[risk_profile]["EQ"]

        per_fi = fi_share / len(FI_TICKERS) if FI_TICKERS else 0.0
        per_eq = eq_share / len(EQ_TICKERS) if EQ_TICKERS else 0.0

        tgt = {**{t: per_fi for t in FI_TICKERS},
               **{t: per_eq for t in EQ_TICKERS}}
    else:
        tgt = (target_weights.to_dict()
               if isinstance(target_weights, pd.Series) else dict(target_weights))

    # ---------- sanitize & normalise ----------------------------------
    tgt = {k: v for k, v in tgt.items() if k in price_data.columns}
    if not tgt:
        raise ValueError("No tickers left after intersecting with price_data")

    sum_weights = sum(tgt.values())
    if not np.isclose(sum_weights, 1.0):
        tgt = {k: v / sum_weights for k, v in tgt.items()}  # re-normalize if needed

    invest = list(tgt.keys())

    # ---------- calendars & cash plan ---------------------------------
    reb_set = set(price_data.resample(rebalance_freq).last().index)
    cash_plan = build_cash_schedule(price_data.index, cash_amount, cash_freq, first_cash_date)
    cash_balance = 0.0

    # ---------- day-0 whole-share purchase ----------------------------
    first_day = price_data.index[0]
    px0 = price_data.loc[first_day]
    hold = {a: int(np.floor(tgt[a] * PV0 / px0[a])) for a in invest}
    cash_balance += PV0 - sum(hold[a] * px0[a] for a in invest)

    rows = []

    # ---------- daily loop --------------------------------------------
    for day in price_data.index:

        cash_balance += cash_plan.loc[day]  # incoming cash
        px = price_data.loc[day]
        pv_pos = sum(hold[a] * px[a] for a in invest)
        pv = pv_pos + cash_balance

        # -------- scheduled rebalance (whole shares + fees) ---------
        tr_sh = {a: 0 for a in invest}
        if day in reb_set:
            cur_val = {a: hold[a] * px[a] for a in invest}
            pv = sum(cur_val.values()) + cash_balance
            des_val = {a: tgt[a] * pv for a in invest}
            cash_tr = {a: des_val[a] - cur_val[a] for a in invest}

            for a in invest:
                tr_sh[a] = int(np.floor(cash_tr[a] / px[a])) if cash_tr[a] > 0 \
                           else int(np.ceil(cash_tr[a] / px[a]))

            dollars = {a: tr_sh[a] * px[a] for a in invest}
            need = sum(d for d in dollars.values() if d > 0)
            recv = -sum(d for d in dollars.values() if d < 0)
            gross = sum(abs(d) for d in dollars.values())
            fee = gross * cost_rate

            cash_balance += recv
            need_total = need + fee

            if need_total > cash_balance:  # scale buys
                scale = cash_balance / need_total
                for a in invest:
                    if tr_sh[a] > 0:
                        tr_sh[a] = int(np.floor(tr_sh[a] * scale))
                dollars = {a: tr_sh[a] * px[a] for a in invest}
                need = sum(d for d in dollars.values() if d > 0)
                gross = sum(abs(d) for d in dollars.values())
                fee = gross * cost_rate
                need_total = need + fee

            cash_balance -= need_total

            # cure small negative cash (residual)
            if cash_balance < 0:
                look30 = np.log(price_data.loc[:day].iloc[-31:].pct_change() + 1).dropna()
                sharpe_by_asset = (look30.mean() / look30.std()).sort_values()
                worst = sharpe_by_asset.index[0]
                while cash_balance < 0 and hold.get(worst, 0) > 0:
                    hold[worst] -= 1
                    cash_balance += px[worst] - px[worst] * cost_rate

            for a in invest:  # update holdings
                hold[a] += tr_sh[a]

        # -------- log ------------------------------------------------
        rec = {"Date": day,
               ("PORTFOLIO", "Value"): pv_pos + cash_balance,
               ("PORTFOLIO", "Cash"): cash_balance}
        for a in invest:
            val = hold[a] * px[a]
            rec[(a, "Weight")] = val / (pv_pos + cash_balance) if pv else 0.0
            rec[(a, "Holdings")] = hold[a]
            rec[(a, "Trade")] = tr_sh[a]
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("Date")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

def compare_performance(port_val: pd.Series,
                        bench_px: pd.Series,
                        freq: int = 252,
                        rf: float = 0.0) -> pd.DataFrame:
    """
    port_val : 'PORTFOLIO | Value' series from your back‑test
    bench_px : daily price of benchmark ticker (same date index)
    """
    port = perf_metrics(port_val,  freq, rf)
    bench= perf_metrics(bench_px,  freq, rf)

    df = (pd.DataFrame({"Portfolio": port, "Benchmark": bench})
            .rename_axis("Metric"))
    return df

def perf_metrics(prices, freq: int = 252, rf: float = 0.0):
    """
    Calcula métricas de rendimiento financiero para una serie de precios o valores de portfolio.

    Args:
        prices: Serie o DataFrame con una sola columna, con precios diarios.
        freq: Frecuencia anualizada (252=daily, 52=weekly...).
        rf: Tasa libre de riesgo por periodo.

    Returns:
        dict con métricas de rendimiento.
    """
    # Convertir a Series si es DataFrame univariado
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            raise ValueError("perf_metrics solo acepta DataFrames con una sola columna.")
    elif not isinstance(prices, pd.Series):
        raise TypeError("Se esperaba una Series o un DataFrame univariado.")

    prices = prices.dropna()

    # daily arithmetic returns
    ret = prices.pct_change().dropna()

    cum_ret = prices.iloc[-1] / prices.iloc[0] - 1
    ann_ret = (1 + cum_ret) ** (freq / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(freq)
    sharpe = (ann_ret - rf * freq) / ann_vol if ann_vol != 0 else np.nan

    # drawdown
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "Cumulative Return": cum_ret,
        "Annualised Return": ann_ret,
        "Annualised Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }

def perf_metrics_v2(prices, cash_flows=None, freq: int = 252, rf: float = 0.0):
    """
    Calcula métricas de rendimiento financiero para una serie de precios o valores de portfolio,
    incluyendo flujos de caja (aportaciones periódicas).

    Args:
        prices: Serie o DataFrame con una sola columna, con valores diarios del portfolio.
        cash_flows: Serie de aportaciones en las mismas fechas que 'prices'.
        freq: Frecuencia de días al año (252 = diario).
        rf: Tasa libre de riesgo por periodo.

    Returns:
        dict con métricas de rendimiento.
    """
    import numpy as np
    import pandas as pd

    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            raise ValueError("perf_metrics solo acepta DataFrames con una sola columna.")
    elif not isinstance(prices, pd.Series):
        raise TypeError("Se esperaba una Series o un DataFrame univariado.")

    prices = prices.dropna()

    if cash_flows is not None:
        cash_flows = cash_flows.reindex(prices.index).fillna(0.0)
        total_invested = cash_flows.sum() + prices.iloc[0]
    else:
        total_invested = prices.iloc[0]
        cash_flows = pd.Series(0.0, index=prices.index)

    final_value = prices.iloc[-1]
    net_gain = final_value - total_invested

    ret = prices.pct_change().dropna()
    cum_ret = final_value / total_invested - 1
    ann_ret = (1 + cum_ret) ** (freq / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(freq)
    sharpe = (ann_ret - rf * freq) / ann_vol if ann_vol != 0 else np.nan

    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "Total Invested (€)": total_invested,
        "Net Gain (€)": net_gain,
        "Cumulative Return": cum_ret,
        "Annualised Return": ann_ret,
        "Annualised Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

def build_cash_flow_series(index, cash_amount: float = 0.0, cash_freq: str = "BME", first_cash_date=None):
    """
    Genera una Serie con los flujos de caja (aportaciones) en las fechas deseadas.

    Args:
        index (pd.DatetimeIndex): Fechas del backtest.
        cash_amount (float): Monto a aportar en cada fecha (positivo).
        cash_freq (str): Frecuencia de aportación. Ej: "BME" (mes), "BQ" (trimestre).
        first_cash_date (datetime or None): Si se quiere iniciar en una fecha específica.

    Returns:
        pd.Series con los flujos de caja alineados con index.
    """
    import pandas as pd

    if cash_amount == 0.0:
        return pd.Series(0.0, index=index)

    # Crear calendario de fechas de aportación
    cash_dates = pd.Series(0.0, index=index)
    resampled_dates = cash_dates.resample(cash_freq).first().index

    if first_cash_date is not None:
        resampled_dates = resampled_dates[resampled_dates >= pd.to_datetime(first_cash_date)]

    cash_dates.loc[cash_dates.index.isin(resampled_dates)] = cash_amount
    return cash_dates

def compare_multiple_performances_styled(
    portfolios: dict,
    bench_px,
    freq: int = 252,
    rf: float = 0.0
) -> pd.io.formats.style.Styler:
    """
    Compara múltiples portfolios y devuelve un resumen formateado bonito.

    Args:
        portfolios (dict): claves = nombres de portfolios, valores = Series o DataFrames de valores.
        bench_px: precios del benchmark, como Series o DataFrame.
        freq (int): número de periodos por año (252 = diario).
        rf (float): tasa libre de riesgo por periodo.

    Returns:
        Styler: tabla formateada con los resultados de performance.
    """
    def to_series(data):
        if isinstance(data, pd.Series):
            return data.dropna()
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                return data.iloc[:, 0].dropna()
            else:
                raise ValueError("DataFrame debe tener solo una columna para convertirlo en Series.")
        else:
            raise TypeError("La entrada debe ser una Series o DataFrame univariado.")

    results = {}

    # Procesar cada portfolio
    for name, port_val in portfolios.items():
        prices = to_series(port_val)
        results[name] = perf_metrics(prices, freq, rf)

    # Procesar el benchmark
    bench_series = to_series(bench_px)
    results["Benchmark"] = perf_metrics(bench_series, freq, rf)

    df = pd.DataFrame(results).rename_axis("Metric")

    # Formateo condicional
    return (
        df.style
        .format({
            col: "{:.2%}" if "Return" in str(metric) or "Volatility" in str(metric) or "Drawdown" in str(metric)
            else "{:.2f}"
            for col in df.columns
            for metric in df.index
        })
        .background_gradient(cmap="YlGnBu", axis=1)
        .set_caption("📈 Portfolio Performance Comparison")
    )

def compare_multiple_performances(
    portfolios: dict,
    bench_px: pd.Series,
    freq: int = 252,
    rf: float = 0.0
) -> pd.DataFrame:
    """
    Compara el rendimiento de múltiples portfolios contra un benchmark.

    Args:
        portfolios (dict): claves = nombres de portfolios, valores = Series de 'PORTFOLIO | Value'.
        bench_px (Series): Precio diario del benchmark.
        freq (int): Número de periodos por año (252 para diario).
        rf (float): Tasa libre de riesgo.

    Returns:
        DataFrame: Métricas de rendimiento comparativas.
    """
    results = {}

    # Evaluar cada portfolio
    for name, port_val in portfolios.items():
        results[name] = perf_metrics(port_val, freq, rf)

    # Evaluar el benchmark
    results["Benchmark"] = perf_metrics(bench_px, freq, rf)

    df = pd.DataFrame(results).rename_axis("Metric")

    return df

def calculate_betas_with_dynamic_benchmark(
    portfolios: dict,
    benchmark_stocks: pd.Series,
    benchmark_bonds: pd.Series,
    portfolio_types: dict
) -> pd.DataFrame:
    """
    Calcula la beta de múltiples portfolios usando el benchmark adecuado según su tipo (FI/EQ).

    Args:
        portfolios (dict): {'nombre': pd.Series de valores de portfolio}
        benchmark_stocks (pd.Series): Precios del benchmark de acciones.
        benchmark_bonds (pd.Series): Precios del benchmark de bonos.
        portfolio_types (dict): {'nombre': 'stocks' o 'bonds'}

    Returns:
        pd.DataFrame: Tabla con las betas de cada portfolio y el benchmark usado.
    """
    import numpy as np
    import pandas as pd

    results = {}

    # Preprocesar benchmarks
    returns_stocks = np.log(benchmark_stocks).diff().dropna()
    returns_bonds = np.log(benchmark_bonds).diff().dropna()

    for name, port_val in portfolios.items():
        # Decide el benchmark según tipo
        benchmark = returns_stocks if portfolio_types[name] == "stocks" else returns_bonds

        # Alinear datos
        data = pd.concat([port_val, benchmark], axis=1).dropna()
        data.columns = ['Portfolio', 'Benchmark']

        portfolio_returns = np.log(data['Portfolio']).diff().dropna()
        aligned_portfolio_returns, aligned_benchmark_returns = portfolio_returns.align(benchmark, join='inner')

        # Cálculo
        covariance = np.cov(aligned_portfolio_returns, aligned_benchmark_returns)[0, 1]
        variance = np.var(aligned_benchmark_returns)

        beta = covariance / variance
        results[name] = {
            "Beta": beta,
            "Benchmark Used": "Stocks" if portfolio_types[name] == "stocks" else "Bonds"
        }

    df_betas = pd.DataFrame.from_dict(results, orient='index')
    df_betas.index.name = 'Portfolio'

    return df_betas

def calculate_betas_auto_classified(
    portfolios: dict,
    benchmark_stocks: pd.Series,
    benchmark_bonds: pd.Series,
    portfolio_weights: dict,
    fi_threshold: float = 0.5  # Si más del 50% en bonos → es "bonds"
) -> pd.DataFrame:
    """
    Calcula la beta de múltiples portfolios usando el benchmark adecuado.
    Clasifica automáticamente si es un portfolio de renta fija o variable según sus pesos.

    Args:
        portfolios (dict): {'nombre': pd.Series de valores del portfolio}
        benchmark_stocks (pd.Series): Precios diarios del benchmark de acciones.
        benchmark_bonds (pd.Series): Precios diarios del benchmark de bonos.
        portfolio_weights (dict): {'nombre': pd.Series de pesos de los activos}
        fi_threshold (float): Umbral para clasificar como "bonds" (default 50%)

    Returns:
        pd.DataFrame: Betas de cada portfolio + benchmark utilizado
    """
    import numpy as np
    import pandas as pd

    results = {}

    # Precalcular returns de benchmarks
    returns_stocks = np.log(benchmark_stocks).diff().dropna()
    returns_bonds = np.log(benchmark_bonds).diff().dropna()

    for name, port_val in portfolios.items():
        # Determinar si el portfolio es de renta fija o variable
        weights = portfolio_weights[name]
        fi_assets = [ticker for ticker in weights.index if ticker.startswith('F') or ticker.startswith('f')]  # asumiendo que bonos empiezan por F
        eq_assets = [ticker for ticker in weights.index if ticker.startswith('E') or ticker.startswith('e')]  # acciones E

        fi_weight = weights[fi_assets].sum() if fi_assets else 0
        eq_weight = weights[eq_assets].sum() if eq_assets else 0

        # Clasificación automática
        portfolio_type = "bonds" if fi_weight >= fi_threshold else "stocks"

        # Seleccionar benchmark
        benchmark = returns_bonds if portfolio_type == "bonds" else returns_stocks

        # Alinear datos
        data = pd.concat([port_val, benchmark], axis=1).dropna()
        data.columns = ['Portfolio', 'Benchmark']

        portfolio_returns = np.log(data['Portfolio']).diff().dropna()
        aligned_portfolio_returns, aligned_benchmark_returns = portfolio_returns.align(benchmark, join='inner')

        # Cálculo
        covariance = np.cov(aligned_portfolio_returns, aligned_benchmark_returns)[0, 1]
        variance = np.var(aligned_benchmark_returns)

        beta = covariance / variance
        results[name] = {
            "Beta": beta,
            "Classification": portfolio_type.title(),  # Bonds o Stocks
            "FI Weight (%)": fi_weight * 100,
            "EQ Weight (%)": eq_weight * 100
        }

    df_betas = pd.DataFrame.from_dict(results, orient='index')
    df_betas.index.name = 'Portfolio'

    return df_betas

def max_sharpe_with_buckets(exp_ret,
                            cov,
                            rf,
                            fi_mask,
                            fi_target,
                            w_min=0.0,
                            w_max=1.0,
                            winsorize_er=True,
                            winsor_limits=(0.05, 0.95)):
    """
    Máxima ratio Sharpe con:
      • Σ w       = 1
      • Σ w_FI    = fi_target   (peso exacto de Renta Fija)
      • w_min ≤ w ≤ w_max       (por defecto 0-1, es decir, sin cortos)

    Parámetros
    ----------
    exp_ret : 1-D array-like
        Rendimientos esperados (mismo orden que cov).
    cov : 2-D array-like
        Matriz de covarianza anualizada.
    rf : float
        Tasa libre de riesgo anual.
    fi_mask : array-like bool
        Máscara booleana (long = n_assets) con True para los activos FI.
    fi_target : float
        Peso total deseado para la cartera de Renta Fija.
    w_min, w_max : float o array-like
        Límite inferior / superior por activo.
    winsorize_er : bool
        Si True se winsoriza `exp_ret` según `winsor_limits`.
    winsor_limits : tuple
        Percentiles inferior / superior para winsorizar.

    Devuelve
    --------
    w_opt : np.ndarray
        Vector de pesos óptimos.
    """

    exp_ret = np.asarray(exp_ret, dtype=float)
    cov = np.asarray(cov, dtype=float)
    fi_mask = np.asarray(fi_mask, dtype=bool)

    # Winsorize opcional
    if winsorize_er:
        exp_ret = winsorize(exp_ret, limits=winsor_limits)

    n = len(exp_ret)

    # Asegurar bounds del mismo tamaño
    if np.isscalar(w_min):
        w_min = np.full(n, w_min)
    if np.isscalar(w_max):
        w_max = np.full(n, w_max)

    bounds = list(zip(w_min, w_max))

    # ---- función objetivo (–Sharpe) -----------------------------------
    def neg_sharpe(w):
        pr = np.dot(w, exp_ret)
        pv = np.sqrt(w @ cov @ w)
        return -(pr - rf) / pv if pv > 0 else 1e6

    # ---- restricciones ------------------------------------------------
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},                 # total 100 %
        {"type": "eq", "fun": lambda w: np.sum(w[fi_mask]) - fi_target},  # cupo FI
    ]

    # ---- optimización --------------------------------------------------
    x0 = np.full(n, 1 / n)                     # inicial: equi-ponderada
    res = sco.minimize(neg_sharpe, x0,
                       method="SLSQP",
                       bounds=bounds,
                       constraints=cons)

    if not res.success:
        raise RuntimeError(f"Optimisation failed → {res.message}")

    return res.x

def get_stock_estimates_df_v3_rb(
        etf_benchmark_df: pd.DataFrame,
        etf_log_returns:   pd.DataFrame,
        benchmark_log_returns: pd.DataFrame,
        rf: float,
        betas: pd.DataFrame,
        frequency: int = 252
) -> pd.DataFrame:
    """
    Calcula Expected Return, Sharpe Ratio y Volatility anualizados para cada
    ETF de Renta Variable usando su benchmark individual.

    Parameters
    ----------
    etf_benchmark_df      : DataFrame con columnas ['Ticker','Benchmark Ticker']
    etf_log_returns       : DataFrame (index = fechas, cols = ETFs)  (log-returns)
    benchmark_log_returns : DataFrame (index = fechas, cols = índices) (log-ret)
    rf                    : tasa libre de riesgo *diaria* (o de la frecuencia
                            usada en etf_log_returns)
    betas                 : DataFrame 1×N con fila 'Beta' y columnas = ETF
    frequency             : nº periodos por año (252 diario, 12 mensual…)

    Returns
    -------
    pd.DataFrame index = Ticker, columnas = ['Expected Return',
                                             'Sharpe Ratio',
                                             'Volatility']
    """
    results = []

    for _, row in etf_benchmark_df.iterrows():
        etf       = row["Ticker"]
        benchmark = row["Benchmark Ticker"]

        # ---------------- comprobaciones -----------------
        if etf not in etf_log_returns.columns:
            print(f"⚠️ {etf} no existe en etf_log_returns")
            continue
        if benchmark not in benchmark_log_returns.columns:
            print(f"⚠️ {benchmark} no existe en benchmark_log_returns")
            continue

        # ------------- alineación de fechas --------------
        etf_ret, bench_ret = etf_log_returns[etf].align(
            benchmark_log_returns[benchmark], join="inner"
        )

        if len(etf_ret) < 2:
            print(f"⚠️ Not enough data for {etf} vs {benchmark}")
            continue

        # ----------------- cálculos CAPM -----------------
        beta = betas.at["Beta", etf]
        mean_bench = bench_ret.mean()

        exp_ret_period = rf + beta * (mean_bench - rf)           # CAPM period
        exp_ret_annual = (1 + exp_ret_period) ** frequency - 1   # anualizado

        vol_period  = etf_ret.std()
        vol_annual  = vol_period * np.sqrt(frequency)

        rf_annual   = (1 + rf) ** frequency - 1
        sharpe      = ((exp_ret_annual - rf_annual) /
                       vol_annual) if vol_annual != 0 else np.nan

        results.append({
            "Ticker":           etf,
            "Expected Return":  exp_ret_annual,
            "Sharpe Ratio":     sharpe,
            "Volatility":       vol_annual,
        })

    return pd.DataFrame(results).set_index("Ticker")

def bck_test_final_DYNAMIC_v10(
    PV0: float,
    price_data: pd.DataFrame,
    rf_annual: float,
    initial_weights: pd.Series | dict,
    FI_TICKERS: list[str],
    EQ_TICKERS: list[str],
    risk_profile: str,
    PROFILE_TARGET: dict,
    etf_benchmark_df: pd.DataFrame,
    etf_log_returns: pd.DataFrame,
    benchmark_log_returns: pd.DataFrame,
    # ----- parámetros opcionales ---------------------------------------
    lookback_days: int = 252,
    min_obs: int = 60,
    opt_freq: str = "BYE",      # optimizar fin de año
    rebalance_freq: str = "BME",# rebalancear fin de mes
    cash_amount: float = 0.0,
    cash_freq: str = "BME",
    first_cash_date=None,
    cost_rate: float = 0.001,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    winsorize_er: bool = True,
    winsorize_limits: tuple = (0.05, 0.95),
    fallback_equal_weight: bool = True,
):
    """
    Back-test dinámico:
    · Optimización anual Max-Sharpe con restricción de buckets FI/EQ según
      `risk_profile` (targets en PROFILE_TARGET).
    · Rebalance mensual a acciones ENTERAS + cash-in + costes de trading.
    · Portafolio mixto bonos / acciones (ETFs).
    Devuelve (histórico_diario, última_ventana_retornos_usada).
 --------------------------------------------------------------------------------------
        - 'BME' : Business Month End (fin de mes laboral)
        - 'BMS' : Business Month Start (inicio de mes laboral)
        - 'BYE' : Business Year End (fin de año laboral)
        - 'BYS' : Business Year Start (inicio de año laboral)
        - 'BQE' or 'BQ' : Business Quarter End (fin de trimestre laboral)
    """

    # ---------- pesos iniciales (normalizados) -------------------------
    tgt = (pd.Series(initial_weights) if isinstance(initial_weights, dict)
           else initial_weights.copy())
    tgt = tgt[tgt.index.isin(price_data.columns)]
    tgt /= tgt.sum()
    invest = list(tgt.index)

    # ---------- calendarios & cash plan --------------------------------
    opt_set  = set(price_data.resample(opt_freq ).last().index)
    reb_set  = set(price_data.resample(rebalance_freq).last().index)
    cash_plan    = build_cash_schedule(price_data.index, cash_amount,
                                       cash_freq, first_cash_date)
    cash_balance = 0.0

    # ---------- compra día-0 (acciones enteras) ------------------------
    first_day = price_data.index[0]
    px0       = price_data.loc[first_day]
    hold      = {a: int(np.floor(tgt[a]*PV0/px0[a])) for a in invest}
    cash_balance += PV0 - sum(hold[a]*px0[a] for a in invest)

    # ---------- variables de ayuda -------------------------------------
    freq     = 252
    rf_day   = (1+rf_annual)**(1/freq) - 1
    fi_target= PROFILE_TARGET[risk_profile]["FI"]

    rows = []
    win_ret = None  # para retornarlo al final si se quiere inspeccionar

    # ================== LOOP DIARIO ====================================
    for i, day in enumerate(price_data.index):

        # ------ cash in -------------------------------------------------
        cash_balance += cash_plan.loc[day]

        px = price_data.loc[day]
        pv_pos = sum(hold[a]*px[a] for a in invest)
        pv     = pv_pos + cash_balance

        # ======= OPTIMIZACIÓN anual ====================================
        if day in opt_set and i > 0:
            start = max(0, i - lookback_days + 1)
            win_ret = np.log(price_data.iloc[start:i+1].pct_change()+1).dropna()

            if len(win_ret) >= min_obs:
                # --- activos disponibles en ventana --------------------
                available = [a for a in invest if a in win_ret.columns]

                # --- RETORNOS log sólo de la ventana ------------------
                ret_win = win_ret[available]

                # --- separa bonos / acciones --------------------------
                bonds_avail  = [a for a in available if a in FI_TICKERS]
                stocks_avail = [a for a in available if a in EQ_TICKERS]

                # ================= 1. ESTIMACIONES BONOS ==============
                est_bonds = (get_bond_estimates_df(
                                ret_win[bonds_avail],
                                risk_free_rate = rf_day,
                                periods_per_year = freq)
                             if bonds_avail else pd.DataFrame())

                # ================= 2. ESTIMACIONES ACCIONES ===========
                if stocks_avail:
                    bench_sub  = etf_benchmark_df[
                        etf_benchmark_df["Ticker"].isin(stocks_avail)
                    ].reset_index(drop=True)

                    # betas solo con las fechas/tickers disponibles
                    stock_rets = ret_win[stocks_avail]
                    betas = get_betas_def_v3(
                        bench_sub,
                        etf_log_returns     = stock_rets,
                        benchmark_log_returns = benchmark_log_returns
                    )

                    est_stocks = get_stock_estimates_df_v3_rb(
                        etf_benchmark_df    = bench_sub,
                        etf_log_returns     = stock_rets,
                        benchmark_log_returns = benchmark_log_returns,
                        rf       = rf_day,
                        betas    = betas,
                        frequency= freq,
                    )
                else:
                    est_stocks = pd.DataFrame()

                # ---------- concat expectativas -----------------------
                expected_df = pd.concat([est_bonds, est_stocks], axis=0)

                if expected_df.shape[0] < 2:
                    print(f"⚠️ {day} – menos de 2 activos con estimaciones; "
                          "se mantiene la asignación previa.")
                else:
                    exp_r = expected_df["Expected Return"].values
                    cov_m = ret_win[expected_df.index].cov()*freq

                    fi_mask = expected_df.index.isin(FI_TICKERS)

                    # --- Optimización Max-Sharpe con bucket FI/EQ -----
                    opt_w = max_sharpe_with_buckets(
                        exp_ret     = exp_r,
                        cov         = cov_m,
                        rf          = rf_annual,
                        fi_mask     = fi_mask,
                        fi_target   = fi_target,
                        w_min       = min_weight,
                        w_max       = max_weight,
                        winsorize_er= winsorize_er,
                        winsor_limits=winsorize_limits,
                    )
                    tgt = dict(zip(expected_df.index, opt_w))

        # ======= REBALANCE mensual (acciones enteras) ===================
        tr_sh = {a: 0 for a in invest}
        if day in reb_set:
            cur_val = {a: hold[a]*px[a] for a in invest}
            pv      = sum(cur_val.values()) + cash_balance
            des_val = {a: tgt.get(a,0)*pv for a in invest}
            cash_tr = {a: des_val[a]-cur_val[a] for a in invest}

            # --- orden entero
            for a in invest:
                tr_sh[a] = (int(np.floor(cash_tr[a]/px[a]))
                            if cash_tr[a]>0
                            else int(np.ceil(cash_tr[a]/px[a])))

            dollars   = {a: tr_sh[a]*px[a] for a in invest}
            need      = sum(d for d in dollars.values() if d>0)
            recv      = -sum(d for d in dollars.values() if d<0)
            gross_not = sum(abs(d) for d in dollars.values())
            fee       = gross_not * cost_rate

            cash_balance += recv
            if need + fee > cash_balance:
                scale = cash_balance / (need + fee) if need+fee > 0 else 0
                for a in invest:
                    if tr_sh[a] > 0:
                        tr_sh[a] = int(np.floor(tr_sh[a]*scale))
                dollars   = {a: tr_sh[a]*px[a] for a in invest}
                need      = sum(d for d in dollars.values() if d>0)
                gross_not = sum(abs(d) for d in dollars.values())
                fee       = gross_not * cost_rate

            cash_balance -= need + fee

            # --- posible cash < 0 ⇒ vender peor Sharpe -----------------
            if cash_balance < 0:
                look30 = np.log(price_data.loc[:day]
                                .iloc[-31:].pct_change()+1).dropna()
                sharpe30 = (look30.mean()/look30.std()).sort_values()
                worst = sharpe30.index[0]
                while cash_balance < 0 and hold.get(worst,0) > 0:
                    hold[worst] -= 1
                    cash_balance += px[worst] - px[worst]*cost_rate

            # --- aplicar trades a holdings -----------------------------
            for a in invest:
                hold[a] += tr_sh[a]

        # ======= LOG ====================================================
        rec = {"Date": day,
               ("PORTFOLIO","Value"): pv_pos + cash_balance,
               ("PORTFOLIO","Cash") : cash_balance}
        for a in invest:
            val = hold[a]*px[a]
            rec[(a,"Weight")]   = val/(pv_pos+cash_balance) if pv else 0
            rec[(a,"Holdings")] = hold[a]
            rec[(a,"Trade")]    = tr_sh[a]
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("Date")
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df

def compare_multiple_performances_v2(
    portfolios: dict,
    bench_px: pd.Series,
    cash_flows: dict = None,
    freq: int = 252,
    rf: float = 0.0
) -> pd.DataFrame:
    """
    Compara el rendimiento de múltiples portfolios contra un benchmark.

    Args:
        portfolios (dict): claves = nombres de portfolios, valores = Series de 'PORTFOLIO | Value'.
        bench_px (Series): Precio diario del benchmark.
        cash_flows (dict): claves = nombres de portfolios, valores = Series de aportes de capital.
        freq (int): Número de periodos por año (252 para diario).
        rf (float): Tasa libre de riesgo.

    Returns:
        DataFrame: Métricas de rendimiento comparativas.
    """
    results = {}

    for name, port_val in portfolios.items():
        results[name] = perf_metrics_v2(port_val, cash_flows=cash_flows, freq=freq, rf=rf)
        
    first_portfolio_name = next(iter(results))
    tot_inv = results[first_portfolio_name]["Total Invested (€)"]

    bench_ret = bench_px.pct_change().fillna(0)
    cum_r_bench = (1 + bench_ret).cumprod() - 1
    net_gain_bench = (1 + cum_r_bench[-1]) * tot_inv - tot_inv
    annualised_ret = (1 + cum_r_bench[-1]) ** (freq / len(bench_ret)) - 1
    annualised_vol = bench_ret.std() * np.sqrt(freq)
    sr_bench = (annualised_ret - rf * freq) / annualised_vol if annualised_vol != 0 else np.nan
    running_max = bench_px.cummax()
    drawdown = (bench_px - running_max) / running_max
    max_dd = drawdown.min()

    results["Benchmark"] = {
        "Total Invested (€)": tot_inv,
        "Net Gain (€)": net_gain_bench,
        "Cumulative Return": cum_r_bench[-1],
        "Annualised Return": annualised_ret,
        "Annualised Vol": annualised_vol,
        "Sharpe Ratio": sr_bench,
        "Max Drawdown": max_dd
    }
    df = pd.DataFrame(results).rename_axis("Metric")

    return df


















