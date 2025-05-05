

def optimize_profiled_portfolio_v3(
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

def optimize_profiled_portfolio_v7(
    log_returns, expected_returns, cov_matrix, rf,
    bonds, stocks, profile_name, profile_allocation,
    min_weight=0.0,
    max_weight=1.0,
    winsorize_er=False,
    winsorize_limits=(0.05, 0.95)
):
    """
    Optimiza una cartera según perfil de riesgo, con control de min/max weight y optional winsorizing de Expected Returns.
    """
    import numpy as np
    import pandas as pd
    import scipy.optimize as sco

    all_assets = bonds + stocks
    assert len(set(all_assets)) == len(all_assets), f"❌ Activos duplicados en {profile_name}"

    mu = expected_returns.loc[all_assets, 'Expected Return'].values / 100
    cov = cov_matrix.loc[all_assets, all_assets]
    n = len(all_assets)

    # Optional winsorizing
    if winsorize_er:
        lower, upper = np.quantile(mu, winsorize_limits)
        mu = np.clip(mu, lower, upper)

    # Bounds: min_weight ≤ weight ≤ max_weight
    bounds = [(min_weight, max_weight) for _ in range(n)]

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
        except Exception:
            return np.inf

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w[bond_idx]) - profile_allocation["FI"]},
        {"type": "eq", "fun": lambda w: np.sum(w[stock_idx]) - profile_allocation["EQ"]},
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

    return {
        "profile": profile_name,
        "weights": weights_series,
        "return": port_ret * 100,
        "volatility": port_vol * 100,
        "sharpe": port_sr
    }

def analyze_profiled_portfolio_v4(
    bonds,
    stocks,
    log_returns,
    cov_matrix,
    expected_returns,
    rf_annual,
    frequency=252,
    min_weight=0.035,
    PROFILE_TARGET = dict  # Default mínimo 3.5%
):
    """
    Evalúa y optimiza carteras por perfil de riesgo.
    Si la optimización falla, genera cartera Equal Weight respetando FI/EQ.
    """

    # Validar activos
    valid_assets = expected_returns.index.tolist()
    valid_bonds = [a for a in bonds if a in valid_assets]
    valid_stocks = [a for a in stocks if a in valid_assets]

    print("Valid assets:", valid_assets)
    print("Valid bonds:", valid_bonds)
    print("Valid stocks:", valid_stocks)

    if len(valid_bonds) + len(valid_stocks) < 2:
        print("❌ Too few valid assets to optimize.")
        return None

    # Subset returns y covarianza
    log_returns = log_returns[valid_assets]
    cov_matrix = cov_matrix.loc[valid_assets, valid_assets]
    expected_returns = expected_returns.loc[valid_assets]

    print("⚙️ Optimizing portfolios by risk profile...")
    results = {}

    for profile_name, alloc in PROFILE_TARGET.items():
        try:
            # 💡 Ajuste dinámico min_weight
            if (alloc["FI"] > 0.85 or alloc["EQ"] > 0.85):
                if alloc["FI"] == 1.0 or alloc["EQ"] == 1.0:
                    min_weight_profile = 0.0
                else:
                    min_weight_profile = 0.01
            else:
                min_weight_profile = min_weight

            # ⚙️ Intentar optimizar
            result = optimize_profiled_portfolio_3(
                log_returns=log_returns,
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                rf=rf_annual,
                bonds=valid_bonds,
                stocks=valid_stocks,
                profile_name=profile_name,
                profile_allocation=alloc,
                min_weight=min_weight_profile
            )
            results[profile_name] = result
            print(f"✅ {profile_name}: OK (Min weight used: {min_weight_profile:.2%})")

        except Exception as e:
            print(f"⚠️ {profile_name}: optimization failed → fallback Equal Weight (FI/EQ) mode")

            # 🎯 Fallback FI/EQ inteligente
            all_assets = valid_bonds + valid_stocks
            n_bonds = len(valid_bonds)
            n_stocks = len(valid_stocks)

            fi_pct = alloc["FI"]
            eq_pct = alloc["EQ"]

            if n_bonds > 0:
                weights_bonds = np.ones(n_bonds) / n_bonds * fi_pct
            else:
                weights_bonds = np.array([])

            if n_stocks > 0:
                weights_stocks = np.ones(n_stocks) / n_stocks * eq_pct
            else:
                weights_stocks = np.array([])

            weights_array = np.concatenate([weights_bonds, weights_stocks])

            mu = expected_returns.loc[all_assets, 'Expected Return'].values / 100
            cov = cov_matrix.loc[all_assets, all_assets]

            port_ret = np.dot(weights_array, mu)
            port_vol = np.sqrt(weights_array @ cov @ weights_array)
            port_sr = (port_ret - rf_annual) / port_vol

            fallback_result = {
                "profile": profile_name,
                "weights": pd.Series(weights_array, index=all_assets),
                "return": port_ret * 100,
                "volatility": port_vol * 100,
                "sharpe": port_sr
            }

            results[profile_name] = fallback_result
            print(f"✅ {profile_name}: Fallback OK (Equal Weight FI/EQ)")

    return results

def analyze_profiled_portfolio_v9(
    bonds,
    stocks,
    log_returns,
    cov_matrix,
    expected_returns,
    rf_annual,
    frequency=252,
    min_weight=0.035,
    max_weight=1.0,
    winsorize_er=True,
    winsorize_limits=(0.05, 0.95),
    PROFILE_TARGET=dict
):
    """
    Nueva versión que además de optimizar,
    guarda expected_returns y cov_matrix por perfil
    para poder generar random portfolios luego.
    """

    import numpy as np
    import pandas as pd

    # Validar activos
    valid_assets = expected_returns.index.tolist()
    valid_bonds = [a for a in bonds if a in valid_assets]
    valid_stocks = [a for a in stocks if a in valid_assets]

    print("Valid assets:", valid_assets)
    print("Valid bonds:", valid_bonds)
    print("Valid stocks:", valid_stocks)

    if len(valid_bonds) + len(valid_stocks) < 2:
        print("❌ Too few valid assets to optimize.")
        return None

    # Subset matrices
    log_returns = log_returns[valid_assets]
    cov_matrix = cov_matrix.loc[valid_assets, valid_assets]
    expected_returns = expected_returns.loc[valid_assets]

    print("⚙️ Optimizing portfolios by risk profile...")
    results = {}

    for profile_name, alloc in PROFILE_TARGET.items():
        try:
            # Ajustar dinámicamente min_weight según perfil
            if (alloc["FI"] > 0.85 or alloc["EQ"] > 0.85):
                if alloc["FI"] == 1.0:
                    min_weight_profile_bonds = 0.0
                    min_weight_profile_stocks = min_weight
                elif alloc["EQ"] == 1.0:
                    min_weight_profile_bonds = min_weight
                    min_weight_profile_stocks = 0.0
                else:
                    min_weight_profile_bonds = min_weight_profile_stocks = 0.01
            else:
                min_weight_profile_bonds = min_weight_profile_stocks = min_weight

            # Crear bounds personalizados
            all_assets = valid_bonds + valid_stocks
            bounds = []
            for asset in all_assets:
                if asset in valid_bonds:
                    bounds.append((min_weight_profile_bonds, max_weight))
                else:
                    bounds.append((min_weight_profile_stocks, max_weight))

            # ⚙️ Llamar optimizador
            result = optimize_profiled_portfolio_v7(
                log_returns=log_returns,
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                rf=rf_annual,
                bonds=valid_bonds,
                stocks=valid_stocks,
                profile_name=profile_name,
                profile_allocation=alloc,
                min_weight=min_weight,
                max_weight=max_weight,
                winsorize_er=winsorize_er,
                winsorize_limits=winsorize_limits
            )

            # ✨ Añadir expected returns y cov_matrix al output para random gen
            result['expected_returns'] = expected_returns
            result['cov_matrix'] = cov_matrix

            results[profile_name] = result
            print(f"✅ {profile_name}: OK")

        except Exception as e:
            print(f"⚠️ {profile_name}: optimization failed → fallback Equal Weight (FI/EQ) mode")

            all_assets = valid_bonds + valid_stocks
            n_bonds = len(valid_bonds)
            n_stocks = len(valid_stocks)

            fi_pct = alloc["FI"]
            eq_pct = alloc["EQ"]

            if n_bonds > 0:
                weights_bonds = np.ones(n_bonds) / n_bonds * fi_pct
            else:
                weights_bonds = np.array([])

            if n_stocks > 0:
                weights_stocks = np.ones(n_stocks) / n_stocks * eq_pct
            else:
                weights_stocks = np.array([])

            weights_array = np.concatenate([weights_bonds, weights_stocks])

            mu = expected_returns.loc[all_assets, 'Expected Return'].values / 100
            cov = cov_matrix.loc[all_assets, all_assets]

            port_ret = np.dot(weights_array, mu)
            port_vol = np.sqrt(weights_array @ cov @ weights_array)
            port_sr = (port_ret - rf_annual) / port_vol

            fallback_result = {
                "profile": profile_name,
                "weights": pd.Series(weights_array, index=all_assets),
                "return": port_ret * 100,
                "volatility": port_vol * 100,
                "sharpe": port_sr,
                "expected_returns": expected_returns,
                "cov_matrix": cov_matrix
            }

            results[profile_name] = fallback_result
            print(f"✅ {profile_name}: Fallback OK (Equal Weight FI/EQ)")

    return results

def summarize_portfolio_results(results, bonds=None, stocks=None):
    """
    Resume en un DataFrame las métricas principales de las carteras optimizadas por perfil de riesgo.
    
    Args:
        results (dict): Output de analyze_profiled_portfolio_vX()
        bonds (list): Lista de tickers de renta fija
        stocks (list): Lista de tickers de renta variable

    Returns:
        pd.DataFrame: Tabla resumen con retornos, volatilidad, Sharpe, n_assets, % FI/EQ
    """
    summary = []

    for profile, res in results.items():
        try:
            weights = res["weights"]
            n_assets = (weights > 1e-4).sum()

            # Cálculo de porcentajes por clase (en %)
            pct_bonds = weights[weights.index.isin(bonds)].sum() * 100 if bonds else 0.0
            pct_stocks = weights[weights.index.isin(stocks)].sum() * 100 if stocks else 0.0

            summary.append({
                "Profile": profile,
                "Return (%)": res["return"],
                "Volatility (%)": res["volatility"],
                "Sharpe Ratio": res["sharpe"],
                "Assets Invested": n_assets,
                "% Bonds": pct_bonds,
                "% Equity": pct_stocks
            })

        except Exception as e:
            print(f"⚠️ Failed to summarize {profile}: {e}")

    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

    return df_summary

def generate_random_portfolios_by_profile_v9(
    bonds,
    stocks,
    log_returns,
    cov_matrix,
    expected_returns,
    rf_annual,
    n_portfolios=10000,
    seed=42,
    initial_min_weight=0.035,
    PROFILE_TARGET = dict
):
    """
    Genera carteras aleatorias perfiladas, balanceando calidad y fallback inteligente si es necesario.
    """

    import numpy as np
    import pandas as pd

    np.random.seed(seed)
    random_results = {}

    # Validar activos
    valid_assets = expected_returns.index.tolist()
    valid_bonds = [a for a in bonds if a in valid_assets]
    valid_stocks = [a for a in stocks if a in valid_assets]

    print("Valid assets:", valid_assets)
    print("Valid bonds:", valid_bonds)
    print("Valid stocks:", valid_stocks)

    if len(valid_bonds) + len(valid_stocks) < 2:
        print("❌ Too few valid assets to generate portfolios.")
        return None

    # Subset returns y covarianza
    log_returns = log_returns[valid_assets]
    cov_matrix = cov_matrix.loc[valid_assets, valid_assets]
    expected_returns = expected_returns.loc[valid_assets]

    all_assets = valid_bonds + valid_stocks

    print("🎲 Generating random portfolios for each risk profile...")

    for profile_name, alloc in PROFILE_TARGET.items():
        try:
            fi_pct = alloc["FI"]
            eq_pct = alloc["EQ"]

            fi_assets = valid_bonds
            eq_assets = valid_stocks
            n_fi = len(fi_assets)
            n_eq = len(eq_assets)

            if n_fi == 0 and fi_pct > 0:
                raise ValueError(f"No valid bond assets for profile {profile_name}")
            if n_eq == 0 and eq_pct > 0:
                raise ValueError(f"No valid stock assets for profile {profile_name}")

            # Subset expected returns y cov_matrix
            selected_assets = fi_assets + eq_assets
            exp_returns_selected = expected_returns.loc[selected_assets, 'Expected Return'].values / 100
            cov_matrix_selected = cov_matrix.loc[selected_assets, selected_assets].values

            # 🔥 Primer intento exigente
            all_weights = []
            port_returns = []
            port_vols = []

            trials = 0
            max_trials = n_portfolios * 3

            while len(all_weights) < n_portfolios and trials < max_trials:
                trials += 1

                w_fi = np.random.dirichlet(np.ones(n_fi)) * fi_pct if n_fi > 0 else np.array([])
                w_eq = np.random.dirichlet(np.ones(n_eq)) * eq_pct if n_eq > 0 else np.array([])
                w_total = np.concatenate([w_fi, w_eq])

                # Aplicar control de min_weight inicial
                if initial_min_weight > 0 and (w_total < initial_min_weight).sum() > 0:
                    continue

                ret = np.dot(w_total, exp_returns_selected)
                vol = np.sqrt(w_total @ cov_matrix_selected @ w_total)

                port_returns.append(ret)
                port_vols.append(vol)
                all_weights.append(w_total)

            # 🔎 ¿Falló el primer intento?
            if len(all_weights) < n_portfolios:
                print(f"⚠️ {profile_name}: fallback → retrying with lower min_weight (1%)")

                all_weights = []
                port_returns = []
                port_vols = []

                trials = 0
                while len(all_weights) < n_portfolios and trials < max_trials:
                    trials += 1

                    w_fi = np.random.dirichlet(np.ones(n_fi)) * fi_pct if n_fi > 0 else np.array([])
                    w_eq = np.random.dirichlet(np.ones(n_eq)) * eq_pct if n_eq > 0 else np.array([])
                    w_total = np.concatenate([w_fi, w_eq])

                    if 0.01 > 0 and (w_total < 0.01).sum() > 0:
                        continue

                    ret = np.dot(w_total, exp_returns_selected)
                    vol = np.sqrt(w_total @ cov_matrix_selected @ w_total)

                    port_returns.append(ret)
                    port_vols.append(vol)
                    all_weights.append(w_total)

                # 🔥 ¿Sigue fallando? → fallback libre
                if len(all_weights) < n_portfolios:
                    print(f"⚠️ {profile_name}: final fallback → generating without min_weight restrictions.")

                    all_weights = []
                    port_returns = []
                    port_vols = []

                    for _ in range(n_portfolios):
                        w_fi = np.random.dirichlet(np.ones(n_fi)) * fi_pct if n_fi > 0 else np.array([])
                        w_eq = np.random.dirichlet(np.ones(n_eq)) * eq_pct if n_eq > 0 else np.array([])
                        w_total = np.concatenate([w_fi, w_eq])

                        ret = np.dot(w_total, exp_returns_selected)
                        vol = np.sqrt(w_total @ cov_matrix_selected @ w_total)

                        port_returns.append(ret)
                        port_vols.append(vol)
                        all_weights.append(w_total)

            random_results[profile_name] = {
                "returns": np.array(port_returns),
                "vols": np.array(port_vols),
                "weights": np.array(all_weights),
                "expected_returns": exp_returns_selected,
                "cov_matrix": pd.DataFrame(cov_matrix_selected, index=selected_assets, columns=selected_assets),
                "rf": rf_annual
            }

            print(f"✅ Random portfolios generated for: {profile_name}")

        except Exception as e:
            print(f"❌ Failed for {profile_name}: {e}")
            random_results[profile_name] = {"error": str(e)}

    return random_results

def generate_random_portfolios_around_v2(
    base_results,
    rf_annual,
    n_portfolios=5000,
    noise_level=0.02,
    seed=42,
    winsorize_er = True,
    winsorize_limits = (0.05,0.95)
):
    """
    Genera carteras aleatorias cercanas a las soluciones óptimas usando los expected returns y la cov_matrix guardadas en base_results.
    
    Args:
        base_results (dict): Output de analyze_profiled_portfolio_v9.
        rf_annual (float): Tasa libre de riesgo anualizada.
        n_portfolios (int): Número de carteras a generar.
        noise_level (float): Nivel de ruido aplicado a los pesos óptimos (default=2%).
        seed (int): Semilla para reproducibilidad.

    Returns:
        dict: Diccionario de carteras aleatorias por perfil.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(seed)
    random_results = {}

    print("\n🎯 Generating random portfolios around optimized solutions...")

    for profile_name, res in base_results.items():
        try:
            print(f"\n🔄 {profile_name}: Generating random portfolios...")

            if ("expected_returns" not in res) or ("cov_matrix" not in res):
                raise ValueError(f"Expected returns or covariance matrix not stored for {profile_name}.")

            base_weights = res["weights"].values
            exp_returns = res["expected_returns"].values / 100  # pasar a decimal
            # 🔥 Aplicar Winsorizing si está activado
            if winsorize_er:
                from scipy.stats.mstats import winsorize
                exp_returns_selected = winsorize(exp_returns, limits=winsorize_limits)
            cov_matrix = res["cov_matrix"].values
            assets = res["weights"].index.tolist()

            all_weights = []
            port_returns = []
            port_vols = []

            for _ in range(n_portfolios):
                noise = np.random.normal(0, noise_level, size=len(base_weights))
                noisy_weights = base_weights + noise

                # Rectificar: pesos negativos a cero
                noisy_weights = np.maximum(noisy_weights, 0)
                noisy_weights /= noisy_weights.sum()  # reescalar a 1

                ret = np.dot(noisy_weights, exp_returns)
                vol = np.sqrt(noisy_weights @ cov_matrix @ noisy_weights)

                port_returns.append(ret)
                port_vols.append(vol)
                all_weights.append(noisy_weights)

            random_results[profile_name] = {
                "returns": np.array(port_returns),
                "vols": np.array(port_vols),
                "weights": np.array(all_weights),
                "expected_returns": exp_returns,
                "cov_matrix": pd.DataFrame(cov_matrix, index=assets, columns=assets),
                "rf": rf_annual
            }

            print(f"✅Random portfolios generated for: {profile_name}")

        except Exception as e:
            print(f"❌Failed for {profile_name}: {e}")
            random_results[profile_name] = {"error": str(e)}

    return random_results

def get_portfolio_recommendation_profiled(user_profile, profiled_results):
    """
    Devuelve la recomendación de portafolio para un perfil de riesgo específico,
    basado en resultados generados por analyze_portfolio_by_risk_profiles().

    Parameters:
    - user_profile: perfil del usuario → e.g., 'Equilibrated'
    - profiled_results: resultados con restricciones de FI/EQ por perfil

    Returns:
    - Diccionario con pesos óptimos y métricas
    """
    valid_profiles = [
        "Defensive", "Conservative", "Cautious", "Equilibrated",
        "Decided", "Daring", "Aggressive"
    ]

    if user_profile not in valid_profiles:
        raise ValueError(f"Perfil de riesgo '{user_profile}' no reconocido.")

    profile_data = profiled_results.get(user_profile)
    
    if not profile_data:
        raise ValueError(f"No hay datos disponibles para el perfil '{user_profile}'.")

    return {
        "profile": user_profile,
        "weights": profile_data["weights"],
        "expected_return": profile_data["return"],
        "expected_volatility": profile_data["volatility"],
        "sharpe_ratio": profile_data["sharpe"],
    }












































