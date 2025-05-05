
# %% [markdown]
# # **ASSUMPTIONS**
# 
# For construction the portfolio I have download two CVS files  `https://etfdb.com/screener/` which cointains a large dataset of different types of ETFs. For simplicity of the project I have choosen just two types of ETFs: **EQUITY** and **FIXED INCOME**. Also they are that Inhave touched most during my business administartion degree, so I could manage them better than any other type. There are more than 3000 thousand ETFs, but after applying some filters, I was left with about **500** ETFs. But the filter has not ended here. When I tried to look for some benchmarks (for the **STOCK ETFs**) from 399 STOCK ETFs there were, only 58 had benchmark available data (at least in the API of **yahoo finance**). Eventhough, the sample size redouces a lot, from an statistical point of view a sample size higher that **30** can be considered acceptable in order to make analysis.
# 
# **Basic assumptions and sample period:**
# 
# 
# The sample used for portfolio construction, optimization, and random portfolio generation consists of historical data ranging from January 2019 to December 2023.
# All return series were computed on a daily basis, and the investment universe was composed exclusively of ETFs, diversified across different asset classes (fixed income and equities).
# The risk-free rate was assumed to be constant and annualized, and all simulations and optimizations were carried out assuming frictionless markets except where explicit trading costs were incorporated.
# 
# # PARTS OF CONSTRUCTING THE PORTFOLIO
# ### **Part 1: ETF Selection**
# 
# In order to build the base portfolios, I implemented a systematic ETF selection process aimed at maximizing the quality of the investable assets and ensuring adequate diversification among them.
# 
# **Approach followed:**
# 
# - First, I filtered the available ETFs by removing those that did not exhibit a positive Expected Return or a positive Sharpe Ratio.  
#   → This step ensures that I only work with assets that have historically delivered positive risk-adjusted returns.
# 
# - Subsequently, I ranked the ETFs using a **composite score** that combines:
#   - the **normalized Sharpe Ratio** (emphasizing the risk-return profile),
#   - the **normalized Expected Return** (capturing the potential for absolute returns), applying a **configurable weight parameter** (`alpha`) to balance their relative importance.
# 
# - To avoid excessive concentration in highly correlated ETFs, I incorporated an additional selection filter:
#   - Assets are selected iteratively, respecting a **maximum allowed correlation threshold** between them.
#   - This correlation limit is adjusted depending on the type of asset (`equity` or `bond`), with stricter thresholds applied to bonds.
# 
# - Furthermore, in the final selection step, I penalized highly volatile assets by incorporating a **volatility penalty** within the composite score, thus encouraging the selection of more stable ETFs.
# 
# Through this process, I succeeded in selecting a set of high-quality ETFs, diversified not only in terms of returns but also across different risk factors, providing a robust foundation for the construction of risk-profiled portfolios.

# %% [markdown]
# 
# ### **Part 2: Initial Portfolio Optimization by Risk Profile**
# 
# Once the diversified universe of ETFs had been selected, I proceeded to optimize the portfolio weights for each risk profile, following a strategy based on maximizing the Sharpe Ratio under specific constraints."
# 
# **Techniques and steps applied:**
# 
# - **Sharpe Ratio Maximization:**  
#   The primary objective of the optimization was to determine the combination of weights that would maximize the expected Sharpe Ratio, thereby maximizing risk-adjusted returns.
# 
# - **Optimization Model:**  
#   - Objective Function: **Minimize the negative** of the Sharpe Ratio.
#   - Optimization Method: Non-linear constrained optimization using the `SLSQP` algorithm.
#   - Inputs:  
#     - Covariance matrix of asset returns (`cov_matrix`),
#     - Expected returns (`expected_returns`),
#     - Annualized risk-free rate (`rf`).
# 
# - **Constraints Applied:**
#   - **Sum of Weights Constraint:**  
#     The sum of all asset weights must equal 1.
#   - **Asset Class Allocation Constraint:**  
#     - The sum of the weights of fixed income (`FI`) assets must match the target fixed income allocation of the risk profile.
#     - The sum of the weights of equity (`EQ`) assets must match the target equity allocation of the risk profile.
#   - **Individual Weight Constraints:**  
#     Each asset's weight must fall within predefined minimum (`min_weight`) and maximum (`max_weight`) bounds to prevent overly concentrated or excessively fragmented portfolios.
# 
# - **Special Treatment for Extreme Profiles:**  
#   For fully fixed income (Defensive) or fully equity (Aggressive) profiles, the minimum weight constraints were dynamically adjusted to allow greater flexibility within the relevant asset class.
# 
# - **Additional Control Over Expected Returns:**  
#   To mitigate the impact of outlier assets, I optionally applied **Winsorization** (`winsorize_er`) to the expected returns, capping extreme values between selected percentiles (e.g., 5th and 95th).
# 
# 
# Through this optimization framework, each risk profile — from Defensive to Aggressive — is provided with a portfolio aligned with its risk tolerance, maximizing its Sharpe Ratio while respecting realistic and controlled diversification and asset allocation constraints.

# %% [markdown]
# ### **Part 3: Generation of Random Portfolios Around Optimized Solutions**
# 
# Once the optimal portfolios for each risk profile were determined, I implemented a second stage focused on robustness analysis by generating a large set of random portfolios around each optimized solution. This process serves to test the stability and quality of the optimized portfolios under small perturbations.
# 
# **Techniques and methodology applied:**
# 
# - **Perturbation of Optimal Weights:**  
#   Starting from the set of optimal weights obtained during the initial optimization, small random perturbations were applied to simulate alternative plausible portfolios.  
#   - The perturbations were drawn from a normal distribution centered at zero, with a standard deviation proportional to a predefined "noise level" parameter (for instance, 0.005%).
#   - After applying noise, the portfolios were re-normalized to ensure that the sum of weights remained equal to 1, and no negative weights were allowed.
# 
# - **Winsorizing of Expected Returns:**  
#   In order to maintain realism and to avoid portfolios driven by outlier expected returns, I applied optional Winsorizing to the expected returns used in this phase.  
#   This technique involved capping extreme values at selected quantiles (e.g., between the 5th and 95th percentiles).
# 
# - **Preservation of Asset Classes:**  
#   Despite the random perturbations, the random portfolios preserved the original separation between asset classes (bonds and stocks), ensuring that the resulting portfolios still respected the fundamental structure of the risk profiles.
# 
# - **Large Sample Generation:**  
#   For each risk profile, thousands of random portfolios (e.g., 10,000) were generated. This large sample size allowed for the creation of an efficient frontier cloud, providing a visual and statistical validation of the positioning of the optimized portfolio relative to other feasible alternatives.
# 
# - **Key Performance Metrics Computed for Each Random Portfolio:**  
#   - Expected return  
#   - Volatility  
#   - Sharpe ratio
# 
# - **Analysis of Capital Market Line (CML):**  
#   For each set of random portfolios, I also computed the Capital Market Line (CML), identifying the random portfolio with the maximum Sharpe ratio and plotting the theoretical risk-return trade-off.
# 
# This random portfolio generation and analysis phase validated the quality of the optimized portfolios, demonstrating whether they were located near the upper boundary of the feasible region (i.e., near the efficient frontier) and whether they offered a superior risk-return trade-off compared to most randomly generated alternatives.

# %% [markdown]
# # CODE_PART: **PORTFOLIO CONSTRUCTION**

# %% [markdown]
# ## Loading Data and Calculating Log Returns
# 
# In this section, I load the stock prices, bond prices, and benchmark stock prices data from CSV files. I then calculate the log returns for the stock prices, bond prices, and benchmark stock prices to prepare for further analysis. Log returns are used as they represent the continuously compounded returns, which is commonly used in financial models.
# 

# %%
# I set the df where all the ETFs are and would be analyzed in order to choose the best ones for our portfolio
df_equity = pd.read_csv(r"C:\Users\macar\OneDrive\Escritorio\FINAL_DEGREE_PROJECT\duplicados\ETFs_equity_limpio.csv")
df_bonds = pd.read_csv(r"C:\Users\macar\OneDrive\Escritorio\FINAL_DEGREE_PROJECT\duplicados\ETFs_bonds_limpio.csv")
df_joined = pd.read_csv(r"C:\Users\macar\OneDrive\Escritorio\FINAL_DEGREE_PROJECT\duplicados\ETFs_equity_bonds_unido.csv")

# I filter the necessary columns
necessary_columns = [
 'Symbol', 'Name', 'Asset Class', 'Assets', 'Avg. Daily Volume', 'Price',
 'ER', '1 Year Returns', '3 Year Returns', '5 Year Returns', 'Expenses',
 ' Standard Deviation', 'Beta', 'Issuer', 'Inception', '# of Holdings',
 '% In Top 10'
]

# I filter the df with only columns I find interesting
df_joined_filt = df_joined[necessary_columns]
df = df_joined_filt.dropna()# I drop tickers with nan values

# %% [markdown]
# ###  **1. Initial Filter**

# %%
# I start by getting the current year using the datetime module
current_year = datetime.now().year  # This stores the current year in the 'current_year' variable

# I ensure that the 'Inception' column in the dataframe is of type datetime
df['Inception'] = df['Inception'].astype('datetime64[ns]')  # Convert 'Inception' column to datetime type
df.loc[:, 'Inception'] = df['Inception'].astype('datetime64[ns]')  # Reaffirm the conversion of 'Inception' to datetime type

# I then filter the dataframe to keep only the rows that satisfy the following conditions:
# - The 'Assets' value must be greater than 100 million
# - The 'Avg. Daily Volume' must be greater than 100,000
# - The 'Inception' year must be at least 10 years ago (i.e., <= current_year - 10)
df_filtered = df[
    (df['Assets'] > 100e6) &  # Filtering by assets greater than 100 million
    (df['Avg. Daily Volume'] > 100000) &  # Filtering by average daily volume greater than 100,000
    (df['Inception'].dt.year <= current_year - 10)  # Filtering by inception year being at least 10 years ago
]


# %% [markdown]
# ### **2. Category clasification (stocks vs bonds)**

# %%
# I begin by creating two categories based on 'Asset Class' in the dataframe: one for stocks and one for bonds
stocks = df_filtered[df_filtered['Asset Class'].str.contains('Equity', case=False, na=False)]  # Selecting rows where 'Asset Class' contains 'Equity'
bonds = df_filtered[df_filtered['Asset Class'].str.contains('Fixed Income|Bond', case=False, na=False)]  # Selecting rows where 'Asset Class' contains 'Fixed Income' or 'Bond'

# I then extract the tickers of stocks and store them as a list
stocks_tickers = stocks['Symbol'].tolist()

# I load the CSV file that contains stock benchmarks and filter it to only include the tickers present in 'stocks_tickers'
stocks_bench_df = pd.read_csv(r"C:\Users\macar\OneDrive\Escritorio\FINAL_DEGREE_PROJECT\stocks_bench_bonds_bench\STOCKS_BENCH.csv")
stocks_bench_df_filt = stocks_bench_df[stocks_bench_df['Ticker'].isin(stocks_tickers)]  # Filtering rows that match stock tickers

# I load another CSV file for benchmark stock returns (Sreturns_BENCH.csv) and set 'Date' as the index
bx_r = pd.read_csv(r"C:\Users\macar\OneDrive\Escritorio\FINAL_DEGREE_PROJECT\stocks_bench_bonds_bench\Sreturns_BENCH.csv", index_col='Date')

# I check which columns are available in 'bx_r'
bx_available = bx_r.columns

# I filter the 'stocks_bench_df_filt' dataframe to only include stocks that have matching benchmarks in 'bx_available'
stocks_bench_available = stocks_bench_df_filt[stocks_bench_df_filt['Benchmark Ticker'].isin(bx_available)]

# I further filter the original 'stocks_bench_df' to include rows where the 'Benchmark Ticker' is in the columns of 'bx_r'
stocks_bench_df_filt = stocks_bench_df[stocks_bench_df['Benchmark Ticker'].isin(bx_r.columns)]

# I update 'stocks_tickers' to only include tickers that are present in the filtered 'stocks_bench_df_filt'
stocks_tickers = stocks_bench_df_filt['Ticker'].tolist()

# I create a boolean series indicating which rows in 'stocks_bench_df' have an available benchmark
s_tf = stocks_bench_df['Benchmark Ticker'].isin(stocks_bench_df_filt['Benchmark Ticker'].tolist())

# I add a new column 'Available' to 'stocks_bench_df' that shows whether the benchmark is available for each stock
stocks_bench_df['Available'] = s_tf

# I filter out the stocks that do not have a matching benchmark from 'stocks_bench_df'
not_bench_stocks_df = (stocks_bench_df.loc[stocks_bench_df['Available'] == False])

# I create a list of benchmark tickers for stocks that do not have an available benchmark
list_of_not_bench = not_bench_stocks_df['Benchmark Ticker'].tolist()


# %%
# I begin by loading the stock prices and bond prices data from CSV files, specifying 'Date' as the index for stocks prices
stocks_prices = pd.read_csv(r"C:\Users\macar\OneDrive\Escritorio\FINAL_DEGREE_PROJECT\duplicados\ETF_STOCK_prices_available_bench.csv", index_col='Date')  # Loading stock prices with 'Date' as the index
stocks_prices_bench = pd.read_csv(r"C:\Users\macar\OneDrive\Escritorio\FINAL_DEGREE_PROJECT\duplicados\ETF_STOCK_prices_bench.csv")  # Loading benchmark stock prices, no specific index
bonds_prices = pd.read_csv(r"C:\Users\macar\OneDrive\Escritorio\FINAL_DEGREE_PROJECT\duplicados\ETF_BONDS_prices.csv")  # Loading bond prices

# I calculate the log returns for the stock prices, bond prices, and benchmark stock prices.
# Log returns are calculated using the formula: log(1 + pct_change()) and then drop any NaN values that result from pct_change() or initial rows
stocks_ret = np.log(1 + stocks_prices.pct_change()).dropna()  # Calculating log returns for stock prices and removing NaN values
bonds_ret = np.log(1 + bonds_prices.pct_change()).dropna()  # Calculating log returns for bond prices and removing NaN values
stocks_ret_bench = np.log(1 + stocks_prices_bench.pct_change()).dropna()  # Calculating log returns for benchmark stock prices and removing NaN values


# %% [markdown]
# ### **3. Beggining of the analysis**

# %% [markdown]
# ## Calculating Betas and Risk-Free Rates, and Estimating Stock and Bond Returns
# 
# In this section, I calculate the betas for the stocks based on the stock returns and benchmark returns using the function `get_betas_def_v3`. I then retrieve the daily and annual risk-free rates for a specified date range. Using the stock and bond returns, the risk-free rates, and the calculated betas, I estimate the returns for both the stocks and bonds using the `get_stock_estimates_df_v3` and `get_bond_estimates_df` functions, respectively.
# 

# %%
# I start by calculating the betas for the stocks using the function 'get_betas_def_v3'. 
# It takes the filtered stock benchmark data, stock returns, and benchmark returns as inputs.
betas = get_betas_def_v3(stocks_bench_df_filt, stocks_ret, bx_r)

# I then retrieve the daily risk-free rate using the 'get_rf' function, specifying the data source and date range for the period.
rf_daily = get_rf('DGS1', key = 'a25b3cdc6cadbc9d0288a8f935e50c67', observation_date='2019-01-01', end_date='2020-01-02', frequency=252)

# I retrieve the annual risk-free rate using the same 'get_rf' function, but with a frequency of 1 to represent yearly data.
rf_annualy = get_rf('DGS1', key = 'a25b3cdc6cadbc9d0288a8f935e50c67', observation_date='2019-01-01', end_date='2020-01-02', frequency=1)

# I use the 'get_stock_estimates_df_v3' function to calculate the stock estimates. 
# It takes several parameters, including the stock benchmark data, stock returns, benchmark returns, 
# daily risk-free rate, betas, and the frequency (252 for daily data).
estimates_stocks = get_stock_estimates_df_v3(
    etf_benchmark_df=stocks_bench_df_filt,
    etf_log_returns=stocks_ret,
    benchmark_log_returns=bx_r,
    rf=rf_daily,
    betas=betas,
    frequency=252
)

# Finally, I calculate the bond estimates using the 'get_bond_estimates_df' function. 
# It requires the bond returns, annual risk-free rate, and the number of periods per year (252 for daily data).
estimates_bonds = get_bond_estimates_df(
    etf_returns_df=bonds_ret,  
    risk_free_rate=rf_annualy,  
    periods_per_year=252 
)

# %% [markdown]
# ## Filtering and Ranking Stocks Based on Expected Return and Sharpe Ratio
# 
# In this section, I filter the stocks based on positive values for both 'Expected Return' and 'Sharpe Ratio'. After filtering, I normalize both columns to a 0-1 scale to ensure they are on the same range. I then calculate a combined score for each stock, where the Sharpe Ratio has a weight of `alpha` and the Expected Return takes the remaining weight. Finally, I rank the stocks based on the combined score in descending order and display the top-ranked stocks.
# 

# %%
# I start by filtering the 'estimates_stocks' dataframe to only include rows where the 'Expected Return' and 'Sharpe Ratio' are both greater than 0.
# I then create a copy of the filtered dataframe to work with.
df_stocks_filtered = estimates_stocks[
    (estimates_stocks['Expected Return'] > 0) &  # Filtering by positive expected return
    (estimates_stocks['Sharpe Ratio'] > 0)  # Filtering by positive Sharpe ratio
].copy()

# I define a weight parameter 'alpha', which I'll use to combine the Sharpe Ratio and Expected Return in the scoring formula
alpha = 0.7

# I optionally normalize the 'Sharpe Ratio' and 'Expected Return' columns to a 0-1 scale for comparison purposes.
# This ensures that both columns have the same scale before combining them into a final score.
df_stocks_filtered['Sharpe Ratio Norm'] = (df_stocks_filtered['Sharpe Ratio'] - df_stocks_filtered['Sharpe Ratio'].min()) / (df_stocks_filtered['Sharpe Ratio'].max() - df_stocks_filtered['Sharpe Ratio'].min())  # Normalizing Sharpe Ratio
df_stocks_filtered['Expected Return Norm'] = (df_stocks_filtered['Expected Return'] - df_stocks_filtered['Expected Return'].min()) / (df_stocks_filtered['Expected Return'].max() - df_stocks_filtered['Expected Return'].min())  # Normalizing Expected Return

# I calculate a combined 'Score' for each stock based on the weighted sum of the normalized Sharpe Ratio and Expected Return.
# The 'alpha' parameter controls the weight of the Sharpe Ratio in the score.
df_stocks_filtered['Score'] = alpha * df_stocks_filtered['Sharpe Ratio Norm'] + (1 - alpha) * df_stocks_filtered['Expected Return Norm']

# I sort the stocks based on their combined score, in descending order, to rank them.
df_stocks_ranked = df_stocks_filtered.sort_values(by='Score', ascending=False)

# I view the top-ranked stocks based on the combined score.
df_stocks_ranked.head()


# %% [markdown]
# Same for the bonds

# %%
# Filter bonds with positive Expected Return and Sharpe Ratio
df_bonds_filtered = estimates_bonds[
    (estimates_bonds['Expected Return'] > 0) &
    (estimates_bonds['Sharpe Ratio'] > 0)
].copy()  

# Define the weight parameter for the score
alpha = 0.7

# Normalize Sharpe Ratio and Expected Return
df_bonds_filtered['Sharpe Ratio Norm'] = (df_bonds_filtered['Sharpe Ratio'] - df_bonds_filtered['Sharpe Ratio'].min()) / (df_bonds_filtered['Sharpe Ratio'].max() - df_bonds_filtered['Sharpe Ratio'].min())
df_bonds_filtered['Expected Return Norm'] = (df_bonds_filtered['Expected Return'] - df_bonds_filtered['Expected Return'].min()) / (df_bonds_filtered['Expected Return'].max() - df_bonds_filtered['Expected Return'].min())

# Create the combined Score
df_bonds_filtered['Score'] = alpha * df_bonds_filtered['Sharpe Ratio Norm'] + (1 - alpha) * df_bonds_filtered['Expected Return Norm']

# Sort the bonds by Score
df_bonds_ranked = df_bonds_filtered.sort_values(by='Score', ascending=False)

# the top-ranked bonds
df_bonds_ranked.head()

# %% [markdown]
# ## Selecting Diversified Stock and Bond ETFs
# 
# In this section, I use the function `select_diversified_etfs_v3` to select a diversified set of stock and bond ETFs. For stocks, I specify a high correlation threshold (0.85) to ensure that the ETFs are sufficiently diversified, along with a volatility penalty factor of 0.02. For bonds, I use a lower correlation threshold (0.4) and a smaller volatility penalty factor of 0.01. I select 5 ETFs in each case and suppress detailed output by setting `verbose` to False.
# 

# %%
# I use the 'select_diversified_etfs_v3' function to select a diversified set of stock ETFs based on certain criteria.
# I pass the ranked stock ETFs, stock returns, the asset class ('equity'), the number of ETFs to select (5), a correlation threshold (0.85),
# a volatility penalty factor (0.02), and set 'verbose' to False to suppress detailed output.
selected_stocks_v3 = select_diversified_etfs_v3(
    ranked_etfs=df_stocks_ranked,  # Ranked stock ETFs
    returns_df=stocks_ret,  # Stock returns dataframe
    asset_class='equity',  # Asset class for stocks
    n_etfs=5,  # Number of ETFs to select
    base_correlation_threshold=0.85,  # Minimum correlation threshold for diversification
    volatility_penalty_factor=0.02,  # Factor to penalize high volatility
    verbose=False  # Suppressing detailed output
)


# I repeat the same process to select a diversified set of bond ETFs, with different parameters for bonds.
# I pass the ranked bond ETFs, bond returns, the asset class ('bond'), the number of ETFs to select (5), a lower correlation threshold (0.4),
# a smaller volatility penalty factor (0.01), and set 'verbose' to False.
selected_bonds_v3 = select_diversified_etfs_v3(
    ranked_etfs=df_bonds_ranked,  # Ranked bond ETFs
    returns_df=bonds_ret,  # Bond returns dataframe
    asset_class='bond',  # Asset class for bonds
    n_etfs=5,  # Number of ETFs to select
    base_correlation_threshold=0.4,  # Minimum correlation threshold for diversification (lower for bonds)
    volatility_penalty_factor=0.01,  # Smaller penalty for volatility in bonds
    verbose=False  # Suppressing detailed output
)



# %% [markdown]
# ## Correlation Matrix of ETF Returns
# 
# In this section, I combine the selected bond and stock ETFs into a final portfolio and calculate their log returns for the specified period. I then compute and visualize the correlation matrix of the returns to understand how the ETFs in the portfolio are related to each other.
# 

# %%
# Combine the selected bond and stock ETFs into a final portfolio
final_portfololio = selected_bonds_v3 + selected_stocks_v3

# Calculate the log returns for the final portfolio over the specified date range
returns_p = get_log_returns(final_portfololio, '2019-01-01', '2020-01-01')

# Compute the correlation matrix of the returns for the final portfolio
corr_mtx = returns_p.corr()

# %% [markdown]
# # **PORTFOLIO OPTIMIZATION**

# %% [markdown]
# ## Getting the portfolio optimal weights for each risk profile

# %%
# Risk-profile bucket targets
PROFILE_TARGET = {
    "Defensive"   : {"FI": 0.90, "EQ": 0.1},
    "Conservative": {"FI": 0.75, "EQ": 0.25},
    "Cautious"    : {"FI": 0.60, "EQ": 0.40},
    "Equilibrated": {"FI": 0.50, "EQ": 0.50},
    "Decided"     : {"FI": 0.30, "EQ": 0.70},
    "Daring"      : {"FI": 0.15, "EQ": 0.85},
    "Aggressive"  : {"FI": 0.05, "EQ": 0.95},
}

# %%
# Select the estimates for the selected bond ETFs by filtering 'estimates_bonds' based on the tickers in 'selected_bonds_v3'
estimates_sel_bonds = estimates_bonds.loc[estimates_bonds.index.isin(selected_bonds_v3), :]

# Select the estimates for the selected stock ETFs by filtering 'estimates_stocks' based on the tickers in 'selected_stocks_v3'
estimates_sel_stocks = estimates_stocks.loc[estimates_stocks.index.isin(selected_stocks_v3), :]

# Calculate the covariance and covariance matrix (excluding benchmark) for the returns of the final portfolio
cov, cov_matrix_no_bench = covariance_matrix(returns_p)

# Annualize the covariance by multiplying by 252 (assuming 252 trading days in a year)
annual_cov = cov * 252

# In this case the difference is that I will get the rf for the period I will do the backtest
rf = get_rf(interval='DGS1', key='a25b3cdc6cadbc9d0288a8f935e50c67', observation_date='2020-01-01', frequency=252)

# Annualized rf
rf_annual = (1 + rf) ** 252 - 1


# %%
# I create a list of dataframes (estimates for selected bonds and stocks) to combine them
dfs = [estimates_sel_bonds, estimates_sel_stocks]

# I concatenate the two dataframes without resetting the index, keeping the original indices intact
expected_returns_combined = pd.concat(dfs)

# I ensure the index is correct by reordering the rows based on the selected bond and stock tickers
expected_returns_combined = expected_returns_combined.loc[selected_bonds_v3 + selected_stocks_v3]

# I multiply the expected returns by 100 to convert them into percentages (if necessary)
expected_returns_combined = expected_returns_combined * 100


# %% [markdown]
# # Analyzing Optimal Portfolios for Different Risk Profiles
# 
# In this section, I perform an analysis of the optimal portfolios for various risk profiles using the **Markowitz approach**. The main objective is to maximize the **Sharpe Ratio** by optimizing the portfolio's allocation between stocks and bonds.
# 
# ## Assumptions
# 
# 1. **Markowitz Mean-Variance Optimization**:
#    - I am using the **Mean-Variance** optimization technique, where the goal is to maximize the **Sharpe Ratio**. This is done by finding the optimal combination of assets that balances expected return and risk (variance), resulting in the most efficient portfolio.
#    
# 2. **Asset Weight Constraints**:
#    - I set a **minimum weight** of 0.0005 and a **maximum weight** of 1 for each asset in the portfolio. This ensures that no asset has too little weight (minimizing the risk of underrepresentation) or too much weight (limiting concentration risk).
# 
# 3. **Minimizing Outliers in Expected Returns**:
#    - To prevent extreme values from distorting the expected return of the portfolio, I apply a **winsorization** process on the expected returns. This involves limiting extreme outliers within a defined range (between the **5th percentile** and the **90th percentile**) to ensure a more stable and reliable expected return.
# 
# 4. **Lookback Period**:
#    - The analysis assumes a **lookback window of one year**, which corresponds to **252 trading days**. This period is typically used to estimate returns, covariance, and other financial metrics, providing a reasonable timeframe for historical performance analysis.
# 
# ## Objective
# 
# The main objective of this analysis is to identify the optimal portfolio allocations for each risk profile based on the above assumptions. By considering a range of risk profiles, I aim to build portfolios that suit different investor preferences for risk and return, all while maximizing the Sharpe Ratio for optimal risk-adjusted returns.
# 
# ## Function Overview
# 
# The `analyze_profiled_portfolio_v9` function is used to perform the optimization for each risk profile. It requires several inputs, including the selected bond and stock ETFs, historical returns data, covariance matrix, expected returns, and the risk-free rate. Additionally, the function applies the **winsorization** technique to minimize the impact of outliers in expected returns and enforces asset weight constraints.
# 
# Here is a breakdown of the inputs to the function:
# - **`bonds`**: The selected bond ETFs.
# - **`stocks`**: The selected stock ETFs.
# - **`log_returns`**: The log returns of the combined portfolio (stocks and bonds).
# - **`cov_matrix`**: The annualized covariance matrix of asset returns.
# - **`expected_returns`**: The combined expected returns of the stocks and bonds.
# - **`rf_annual`**: The annualized risk-free rate.
# - **`min_weight`**: The minimum allowable weight for each asset.
# - **`max_weight`**: The maximum allowable weight for each asset.
# - **`winsorize_er`**: A flag indicating whether to winsorize the expected returns to minimize the impact of outliers.
# - **`winsorize_limits`**: The percentile limits for winsorization (default is between 5% and 90%).
# - **`PROFILE_TARGET`**: The specific target profile for portfolio optimization.
# 
# By using this approach, I aim to generate portfolios that align with different risk profiles, maximizing risk-adjusted returns.
# 

# %%
results_v9 = analyze_profiled_portfolio_v9(
    bonds=selected_bonds_v3,
    stocks=selected_stocks_v3,
    log_returns=returns_p,
    cov_matrix=annual_cov,
    expected_returns=expected_returns_combined,
    rf_annual=rf_annual,
    min_weight=0.0005,
    max_weight=1,
    winsorize_er=True,
    winsorize_limits=(0.05,0.9),
    PROFILE_TARGET=PROFILE_TARGET
    )

# %% [markdown]
# The table below displays the results of portfolio optimization for different risk profiles. Each row represents a different profile, with key metrics such as return, volatility, Sharpe Ratio, asset allocation, and the percentage of bonds and equity invested.
# The optimization appears to be in order, as the weights in bonds and equity for each profile respect the expected risk-return trade-off. More conservative profiles allocate more to bonds, while more aggressive profiles favor equities, in line with the profile labels. The other financial metrics, such as return, volatility, and Sharpe Ratio, also reflect these adjustments appropriately.

# %%
summary_df = summarize_portfolio_results(results_v9, bonds=selected_bonds_v3, stocks=selected_stocks_v3)
summary_df

# %% [markdown]
# ## Graphical plots to see the distribution of the weights for the different risk profiles

# %% [markdown]
# ## Generating Random Portfolios for Efficient Frontier Visualization
# 
# In this section, I generate a large number of random portfolios (100,000) around the base results of the optimal portfolios for each risk profile. This is done by adding small noise to the base portfolio allocations. The generated random portfolios will be used to plot the efficient frontier and illustrate the concept of efficient portfolios as per the Markowitz model. I also apply winsorization to the expected returns to mitigate the impact of outliers and ensure a more stable representation of the portfolios.
# 

# %% [markdown]
# In the plots there are three modes for generating and plotting random portfolios with respect to their returns and volatilities: **`historical`**, **`noisy`**, and **`mixed`**. Each of these modes represents a different approach to the data used for the portfolios. Let's break down the meaning of each mode:
# 
# ### 1. **Historical Mode**:
#    - **Description**: This mode uses the actual historical returns of the assets (stocks, bonds, etc.) over a specific period of time. It assumes that past performance is indicative of future performance, and portfolios are constructed based on the historical data (both returns and volatility) of the selected assets.
#    - **Use Case**: This mode is useful when you want to analyze the performance of random portfolios based strictly on historical data. It provides a reflection of how the portfolios would have performed in the past.
# 
# ### 2. **Noisy Mode**:
#    - **Description**: The "noisy" mode introduces random noise to the returns of the assets. This means that the portfolio construction incorporates random fluctuations or errors in the data, making the return series more volatile and unpredictable than the actual historical data.
#    - **Use Case**: This mode can be useful when you want to test the robustness of a portfolio's performance under uncertain or fluctuating market conditions. It simulates the effect of market unpredictability and noise on portfolio performance.
# 
# ### 3. **Mixed Mode**:
#    - **Description**: The "mixed" mode combines both historical and noisy returns. This mode typically uses historical data as a base but introduces some noise or randomness to simulate a more dynamic and uncertain environment. It could involve adjusting the returns by adding a controlled level of noise to reflect a mix of past performance and potential market changes.
#    - **Use Case**: The mixed mode provides a more balanced view by accounting for both past performance and market uncertainty. It’s useful for evaluating portfolio performance in a more realistic context where markets are never entirely predictable.
# 
# These modes are selected based on the desired approach for simulating portfolio performance under different market scenarios. In the context of your analysis, you might use each mode to visualize how portfolios would behave under different assumptions of market stability or uncertainty.
# 
# The one that has worked best for me for plotting the random portfolios is `Historical` mode.

# %% [markdown]
# We can see that the results align very closely between the actual estimates (table) and the random portfolios (plot). The similarities in the performance metrics—such as the return, volatility, and Sharpe ratio—suggest that the random portfolios, while introducing some noise, closely mirror the behavior of the optimized portfolios based on historical data.
# 
# ### Explanation of the Small Differences:
# The small differences between the two sets of results can be attributed to the nature of the **random noise** introduced in the generation of the random portfolios. When we generate random portfolios, the idea is to add slight variations around the base results (from the historical data) to simulate different scenarios. The slight discrepancies could be due to:
# - **Noise in the random portfolios**: Random fluctuations were added to the weights and returns in the random portfolios, leading to slight deviations from the historical or "optimized" returns.
# - **Slight imperfections in the random generation process**: Although the random portfolios are generated to reflect a similar distribution of returns, the exact configuration might vary due to the randomness inherent in the process.
# Overall, the random portfolios give us a more dynamic view of possible portfolio outcomes under varying conditions, but they remain very close to the actual estimates, indicating that the base optimization is a good predictor of future performance in similar market conditions.


