


def plot_portfolio_summary_pro(summary_df):
    """
    Gráfico de barras elegante para comparar Retorno (%) y Volatilidad (%) por perfil de riesgo.

    Args:
        summary_df (DataFrame): Output de summarize_portfolio_results()
    """

    profiles = summary_df["Profile"]
    returns = summary_df["Return (%)"]
    vols = summary_df["Volatility (%)"]

    x = np.arange(len(profiles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    # Barras con colores suaves
    bars1 = ax.bar(x - width/2, returns, width, label='Return (%)', color="#4CAF50", alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, vols, width, label='Volatility (%)', color="#2196F3", alpha=0.8, edgecolor='black')

    # Líneas guía horizontales
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)

    # Títulos y etiquetas
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Risk Profile', fontsize=12)
    ax.set_title('Return vs Volatility by Risk Profile', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)

    # Añadir valores encima de las barras
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)

    # Estética final
    plt.tight_layout()
    plt.show()

def plot_selected_etfs_correlation(selected_etfs, returns_df, title="Selected ETFs Correlation Matrix"):
    """
    Plots a heatmap of the correlation matrix for the selected ETFs.
    
    Args:
        selected_etfs (list): List of selected ETF tickers.
        returns_df (pd.DataFrame): DataFrame of returns (columns = ETFs).
        title (str): Plot title.
    """
    selected_returns = returns_df[selected_etfs]
    correlation_matrix = selected_returns.corr()
    
    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=16)
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", square=True, cbar=True)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_risk_return_scatter_pro(summary_df):
    """
    Scatter plot elegante: Volatilidad (%) vs Retorno (%) para perfiles de riesgo.

    Args:
        summary_df (DataFrame): Output de summarize_portfolio_results()
    """

    plt.figure(figsize=(12, 7))
    
    # Scatter
    plt.scatter(summary_df['Volatility (%)'], summary_df['Return (%)'],
                s=200, c='#2196F3', edgecolors='black', alpha=0.85, marker='o')

    # Anotar perfiles
    for i, profile in enumerate(summary_df['Profile']):
        plt.annotate(profile,
                     (summary_df['Volatility (%)'][i], summary_df['Return (%)'][i]),
                     textcoords="offset points",
                     xytext=(8,5),
                     ha='left',
                     fontsize=10,
                     fontweight='bold')

    # Ejes y grid
    plt.xlabel('Volatility (%)', fontsize=13)
    plt.ylabel('Return (%)', fontsize=13)
    plt.title('Risk vs Return by Risk Profile', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Mejorar límites para que no estén pegados
    x_margin = (summary_df['Volatility (%)'].max() - summary_df['Volatility (%)'].min()) * 0.15
    y_margin = (summary_df['Return (%)'].max() - summary_df['Return (%)'].min()) * 0.15
    plt.xlim(summary_df['Volatility (%)'].min() - x_margin, summary_df['Volatility (%)'].max() + x_margin)
    plt.ylim(summary_df['Return (%)'].min() - y_margin, summary_df['Return (%)'].max() + y_margin)

    plt.tight_layout()
    plt.show()

def plot_risk_profile_subplot_no_EFL_v2(ax, name, data, rf_rate):
    port_returns = data['returns']
    port_vols = data['vols']
    weights = data['weights']
    exp_returns = data['expected_returns']
    cov_matrix = data['cov_matrix']

    max_sr_ret, max_sr_vol, max_sr_weights, max_sr, sharpe_ratios = max_sharpe_ratio(
        port_returns, rf_rate, port_vols, weights
    )

    # Fondo blanco puro
    ax.set_facecolor('white')

    # Scatter de portafolios aleatorios
    scatter = ax.scatter(
        port_vols, port_returns, 
        c=sharpe_ratios, cmap="viridis", 
        edgecolors='k', linewidths=0.3, alpha=0.8, s=30
    )

    # CML
    extended_vol = max_sr_vol * 1.2
    cml_x = np.linspace(0, extended_vol, 500)
    slope = (max_sr_ret - rf_rate) / max_sr_vol
    cml_y = rf_rate + slope * cml_x
    ax.plot(
        cml_x, cml_y, color='red', linestyle='-', linewidth=2.5, 
        label='Capital Market Line'
    )

    # Max Sharpe Point
    ax.scatter(
        max_sr_vol, max_sr_ret, 
        color='gold', edgecolors='black', marker='*', 
        s=300, label='Max Sharpe', zorder=5
    )

    # Ajustar los límites del gráfico de manera inteligente
    padding_x = (port_vols.max() - port_vols.min()) * 0.15
    padding_y = (port_returns.max() - port_returns.min()) * 0.15
    ax.set_xlim(port_vols.min() - padding_x, port_vols.max() + padding_x)
    ax.set_ylim(port_returns.min() - padding_y, port_returns.max() + padding_y)

    # 🧾 Texto resumen de retorno, volatilidad y Sharpe ratio
    ax.text(0.02, 0.95, 
            f"Return: {max_sr_ret*100:.2f}%\nVolatility: {max_sr_vol*100:.2f}%\nSharpe: {max_sr:.2f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # Estética general
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_xlabel("Volatility", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

def plot_all_profiles_subplots_noEFL_v2(random_results, rf):

    n = len(random_results)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    for i, (name, data) in enumerate(random_results.items()):
        plot_risk_profile_subplot_no_EFL_v2(axes[i], name, data, rf)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Random Portfolios + CML", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def plot_single_risk_profile(random_results, profile_name, rf_rate):
    """
    Plotea solo un perfil de riesgo seleccionado con portafolios aleatorios + CML.

    Args:
        random_results (dict): Output de generate_random_portfolios_by_profile.
        profile_name (str): Nombre del perfil que se quiere graficar (ej: 'Cautious').
        rf_rate (float): Tasa libre de riesgo anualizada.
    """

    if profile_name not in random_results:
        print(f"❌ Profile {profile_name} not found in results.")
        return

    data = random_results[profile_name]

    fig, ax = plt.subplots(figsize=(10, 6))

    port_returns = data['returns']
    port_vols = data['vols']
    weights = data['weights']
    exp_returns = data['expected_returns']
    cov_matrix = data['cov_matrix']

    # Calcular Max Sharpe
    max_sr_ret, max_sr_vol, max_sr_weights, max_sr, sharpe_ratios = max_sharpe_ratio(
        port_returns, rf_rate, port_vols, weights
    )

    # Fondo blanco
    ax.set_facecolor('white')

    # Scatter de carteras aleatorias
    scatter = ax.scatter(
        port_vols, port_returns, 
        c=sharpe_ratios, cmap="viridis", 
        edgecolors='k', linewidths=0.3, alpha=0.8, s=40
    )

    # Capital Market Line (CML)
    extended_vol = max_sr_vol * 1.2
    cml_x = np.linspace(0, extended_vol, 500)
    slope = (max_sr_ret - rf_rate) / max_sr_vol
    cml_y = rf_rate + slope * cml_x
    ax.plot(cml_x, cml_y, color='red', linestyle='-', linewidth=2.5, label='Capital Market Line')

    # Max Sharpe Point
    ax.scatter(
        max_sr_vol, max_sr_ret,
        color='gold', edgecolors='black', marker='*',
        s=300, label='Max Sharpe', zorder=5
    )

    # Ajuste de límites inteligentes
    padding_x = (port_vols.max() - port_vols.min()) * 0.15
    padding_y = (port_returns.max() - port_returns.min()) * 0.15
    ax.set_xlim(port_vols.min() - padding_x, port_vols.max() + padding_x)
    ax.set_ylim(port_returns.min() - padding_y, port_returns.max() + padding_y)

    # Texto resumen
    ax.text(0.02, 0.95, 
            f"Return: {max_sr_ret*100:.2f}%\nVolatility: {max_sr_vol*100:.2f}%\nSharpe: {max_sr:.2f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # Etiquetas
    ax.set_title(f"Random Portfolios - {profile_name} Profile", fontsize=15, fontweight='bold')
    ax.set_xlabel("Volatility", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_multiple_portfolio_values_plus(portfolio_dict, title="Portfolio Value Over Time"):
    """
    Plotea múltiples portfolios de valor en un único gráfico.
    Añade el retorno total (%) en la leyenda.

    Args:
        portfolio_dict (dict): 
            claves = nombres de portfolios (str),
            valores = Series o DataFrames con Portfolio Value.
        title (str): Título del gráfico.
    """
    plt.figure(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(portfolio_dict)))

    for i, (name, portfolio_series) in enumerate(portfolio_dict.items()):
        # Extraer serie de valor
        if isinstance(portfolio_series, pd.DataFrame):
            series_to_plot = portfolio_series[("PORTFOLIO", "Value")]
        else:
            series_to_plot = portfolio_series

        # Calcular retorno acumulado
        initial_value = series_to_plot.iloc[0]
        final_value = series_to_plot.iloc[-1]
        total_return = (final_value / initial_value - 1) * 100

        # Graficar
        plt.plot(
            series_to_plot,
            color=colors[i],
            linewidth=1.5,
            linestyle='-',
            label=f"{name} ({total_return:+.1f}%)"
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (€)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_portfolio_pie_chart_pro(recommendation):
    weights = recommendation["weights"]
    labels = weights.index
    values = weights.values * 100

    # Colores personalizados
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # Estilo de texto
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('black')

    ax.set_title(f"Recommended Portfolio: {recommendation['profile']}", fontsize=15, fontweight='bold')
    plt.axis('equal')  # Mantener círculo
    plt.tight_layout()
    plt.show()

def plot_portfolio_bar_chart_pro(recommendation):
    weights = recommendation["weights"]
    tickers = weights.index
    values = weights.values * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        tickers, values, 
        color='dodgerblue', edgecolor='black'
    )

    ax.set_xlabel("Weight (%)", fontsize=12)
    ax.set_title(f"Recommended Portfolio: {recommendation['profile']}", fontsize=15, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Añadir valores al final de las barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_risk_profile_subplot_no_EFL_v3(ax, name, data, rf_rate, mode='mixed'):
    """
    Nuevo plot adaptado a random portfolios con 3 tipos de retornos/vols: historical, noisy, mixed.
    """
    port_returns_all = data['returns']
    port_vols_all = data['vols']
    weights = data['weights']
    exp_returns = data['expected_returns']
    cov_matrix = data['cov_matrix']

    # 🔥 Manejar casos en que NO son tuplas
    if isinstance(port_returns_all, (tuple, list)): 
        mode_dict = {'historical': 0, 'noisy': 1, 'mixed': 2}
        idx = mode_dict.get(mode, 2)  # Por defecto mixed
        port_returns = port_returns_all[idx]
        port_vols = port_vols_all[idx]
    else:
        port_returns = port_returns_all
        port_vols = port_vols_all

    # 🔵 Nueva corrección si port_returns tiene más de 1 columna:
    if port_returns.ndim == 2:
        mode_dict = {'historical': 0, 'noisy': 1, 'mixed': 2}
        idx = mode_dict.get(mode, 2)
        port_returns = port_returns[:, idx]

    # Calcular Max Sharpe
    max_sr_ret, max_sr_vol, max_sr_weights, max_sr, sharpe_ratios = max_sharpe_ratio(
        port_returns, rf_rate, port_vols, weights
    )

    # Plot como antes
    ax.set_facecolor('white')
    scatter = ax.scatter(
        port_vols, port_returns, 
        c=sharpe_ratios, cmap="viridis", 
        edgecolors='k', linewidths=0.3, alpha=0.8, s=30
    )

    extended_vol = max_sr_vol * 1.2
    cml_x = np.linspace(0, extended_vol, 500)
    slope = (max_sr_ret - rf_rate) / max_sr_vol
    cml_y = rf_rate + slope * cml_x
    ax.plot(
        cml_x, cml_y, color='red', linestyle='-', linewidth=2.5, 
        label='Capital Market Line'
    )

    ax.scatter(
        max_sr_vol, max_sr_ret, 
        color='gold', edgecolors='black', marker='*', 
        s=300, label='Max Sharpe', zorder=5
    )

    padding_x = (port_vols.max() - port_vols.min()) * 0.15
    padding_y = (port_returns.max() - port_returns.min()) * 0.15
    ax.set_xlim(port_vols.min() - padding_x, port_vols.max() + padding_x)
    ax.set_ylim(port_returns.min() - padding_y, port_returns.max() + padding_y)

    ax.text(0.02, 0.95, 
            f"Return: {max_sr_ret*100:.2f}%\nVolatility: {max_sr_vol*100:.2f}%\nSharpe: {max_sr:.2f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(f"{name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Volatility", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

def plot_all_profiles_subplots_noEFL_v3(random_results, rf, mode='mixed'):
    n = len(random_results)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(17, 5 * rows))
    axes = axes.flatten()

    for i, (name, data) in enumerate(random_results.items()):
        plot_risk_profile_subplot_no_EFL_v3(axes[i], name, data, rf, mode=mode)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Random Portfolios + CML", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def calculate_and_plot_portfolio_beta(portfolio_value, benchmark_value, title="Portfolio vs Benchmark"):
    """
    Calcula y grafica la beta de un portfolio respecto a un benchmark.

    Args:
    - portfolio_value (pd.Series): Valor del portfolio a lo largo del tiempo.
    - benchmark_value (pd.Series): Valor del benchmark a lo largo del tiempo.

    Returns:
    - beta (float): Beta del portfolio respecto al benchmark.
    """

    # Asegurar índices compatibles
    data = pd.concat([portfolio_value, benchmark_value], axis=1).dropna()
    data.columns = ['Portfolio', 'Benchmark']

    # Calcular log-returns
    port_ret = np.log(data['Portfolio']).diff().dropna()
    bench_ret = np.log(data['Benchmark']).diff().dropna()

    # Alinear por seguridad
    port_ret, bench_ret = port_ret.align(bench_ret, join='inner')

    # Ajuste lineal: port_ret ~ beta * bench_ret + alpha
    slope, intercept, r_value, p_value, std_err = linregress(bench_ret, port_ret)
    beta = slope

    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(bench_ret, port_ret, alpha=0.5)
    x = np.linspace(bench_ret.min(), bench_ret.max(), 100)
    plt.plot(x, intercept + slope * x, color='red', label=f'Beta = {beta:.3f}')
    plt.title(title)
    plt.xlabel('Benchmark Returns')
    plt.ylabel('Portfolio Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

    return beta

def calculate_portfolio_beta(portfolio_value, benchmark_value):
    """
    Calcula la beta de un portfolio respecto a un benchmark.

    Args:
    - portfolio_value (pd.Series): Valor del portfolio a lo largo del tiempo.
    - benchmark_value (pd.Series): Valor del benchmark a lo largo del tiempo.

    Returns:
    - beta (float): Beta del portfolio respecto al benchmark.
    """
    import numpy as np
    import pandas as pd

    # Asegurar que los índices coinciden
    data = pd.concat([portfolio_value, benchmark_value], axis=1).dropna()
    data.columns = ['Portfolio', 'Benchmark']

    # Calcular log-returns
    portfolio_returns = np.log(data['Portfolio']).diff().dropna()
    benchmark_returns = np.log(data['Benchmark']).diff().dropna()

    # Alinear por seguridad
    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')

    # Covarianza y varianza
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)

    # Beta
    beta = covariance / variance

    return beta

def plot_portfolio_statistics(
    dynamic_df: pd.DataFrame = None,
    static_df: pd.DataFrame = None,
    col: tuple = ("PORTFOLIO", "Value"),
    title: str = "Portfolio Value Over the Years"
):
    """
    Dibuja los valores históricos de uno o dos portfolios (dinámico y/o estático).

    Args:
        dynamic_df (DataFrame, opcional): DataFrame del portfolio dinámico.
        static_df (DataFrame, opcional): DataFrame del portfolio estático.
        col (tuple): Tuple de MultiIndex que indica la columna con los valores.
        title (str): Título del gráfico.

    Raises:
        ValueError: Si no se proporciona ningún DataFrame.
    """
    if dynamic_df is None and static_df is None:
        raise ValueError("Debes proporcionar al menos un DataFrame: dynamic_df o static_df.")

    plt.figure(figsize=(12, 6))

    if dynamic_df is not None:
        plt.plot(
            dynamic_df[col],
            color='royalblue',
            linewidth=0.5,
            linestyle='-',
            marker='o',
            markersize=4,
            label='Portfolio Value DYNAMIC'
        )

    if static_df is not None:
        plt.plot(
            static_df[col],
            color='red',
            linewidth=0.5,
            linestyle='-',
            marker='o',
            markersize=4,
            label='Portfolio Value STATIC'
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value (€)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_tickers_columns(
    df: pd.DataFrame,
    tickers: list,
    columns: list,
    start_date: str = None,
    end_date: str = None,
    title_prefix: str = "Time Series of",
    figsize: tuple = (14, 6),
    linewidth: float = 2.0,
    linestyle: str = '-'
):
    """
    Plotea series temporales de columnas específicas para tickers seleccionados.

    Args:
        df (pd.DataFrame): DataFrame con MultiIndex en columnas (nivel 0 = ticker, nivel 1 = métrica).
        tickers (list): Lista de tickers a mostrar (ej. ['AAPL', 'MSFT']).
        columns (list): Lista de columnas a graficar (ej. ['Close', 'Volume']).
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'. Opcional.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'. Opcional.
        title_prefix (str): Prefijo para el título del gráfico.
        figsize (tuple): Tamaño del gráfico.
        linewidth (float): Grosor de las líneas.
        markersize (int): Tamaño de los puntos.
        linestyle (str): Estilo de línea (por defecto '-').
    """
    # Filtrar fechas si se especifican
    df_filtered = df.copy()
    if start_date:
        df_filtered = df_filtered[df_filtered.index >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered.index <= pd.to_datetime(end_date)]

    for col in columns:
        plt.figure(figsize=figsize)
        plotted = False

        for ticker in tickers:
            try:
                serie = df_filtered[(ticker, col)].dropna()
                plt.plot(
                    serie,
                    label=ticker,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )
                plotted = True
            except KeyError:
                print(f"⚠️  No se encontró '{col}' para '{ticker}'. Lo omito.")
                continue

        if plotted:
            plt.title(f"{title_prefix} {col}", fontsize=16, fontweight='bold')
            plt.xlabel("Date", fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend(title="Ticker")
            plt.tight_layout()
            plt.show()
        else:
            print(f"No se pudo graficar ninguna serie para la métrica '{col}'.")

def plot_portfolio_dist_v2(horizon_days=22):
    """
    horizon_days : número de días hábiles hacia adelante (22≈1 mes, 252≈1 año)
    """
    # Asegurar alineación temporal
    pv = pv_series.copy()
    cash = cash_series.reindex(pv.index).fillna(0)
    
    # Cálculo de ganancia neta acumulada diaria
    total_invested_series = cash.cumsum()
    net_gain_series = pv - total_invested_series
    net_gain_series = net_gain_series.dropna()

    # Media y desviación diaria de la ganancia neta
    daily_net_gain_diff = net_gain_series.diff().dropna()
    mu_d = daily_net_gain_diff.mean()
    sigma_d = daily_net_gain_diff.std()

    # Ajustar al horizonte
    mu_p = mu_d * horizon_days
    sigma_p = sigma_d * np.sqrt(horizon_days)

    # Ejes para la curva normal
    x = np.linspace(mu_p - 2.2*sigma_p, mu_p + 2.2*sigma_p, 200)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, norm.pdf(x, mu_p, sigma_p), lw=2)

    # Annotate average gain
    ax.annotate(f"Average\nGain\n${mu_p:,.0f}",
                xy=(mu_p, norm.pdf(mu_p, mu_p, sigma_p)),
                xytext=(mu_p,
                        norm.pdf(mu_p, mu_p, sigma_p)*0.5),
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->"))

    # Annotate 95% worst-case loss
    wc = mu_p - 1.645 * sigma_p  # one-tailed 95%
    ax.annotate(f"95% Worst‑case\nLoss\n${wc:,.0f}",
                xy=(wc, norm.pdf(wc, mu_p, sigma_p)),
                xytext=(wc,
                        norm.pdf(mu_p, mu_p, sigma_p)*1.2),
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->"))

    ax.set_title(f"{horizon_days}-Day Forward P/L Distribution")
    ax.set_xlabel("Profit / Loss ($)")
    ax.set_ylabel("Density")
    ax.grid(True)
    plt.show()








