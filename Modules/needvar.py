# %%
from Modules.plots_fx import *
from Modules.portfolioconstruction_fx import *
from Modules.score_fx import *
#warnings.filterwarnings('ignore')
from Modules.portfolio_variables import *
# %%
#recommendation = get_portfolio_recommendation_profiled(user_risk_profile, results_v9)
# Filter the benchmark data to include only the selected stocks from the portfolio
stocks_bench_df_filt_2 = stocks_bench_df_filt[stocks_bench_df_filt['Ticker'].isin(selected_stocks_v3)]

# Filter the stock returns for the selected stocks in the portfolio
stocks_ret_filt = stocks_ret[selected_stocks_v3]

# Get the list of benchmark tickers corresponding to the selected stocks
selected_bx = stocks_bench_df_filt_2['Benchmark Ticker'].tolist()

# Filter the benchmark returns data to include only the selected benchmark tickers
bx_r_filt = bx_r[selected_bx]

# Retrieve historical stock price data for the selected assets over the test period (2020-01-01 to 2025-04-30)
prices_df = get_multiple_stocks_data(final_portfololio, '2020-01-01', '2025-04-30')

# Get the percentage change (returns) for the selected stocks over the testing period
stocks_ret_filt_v2 = get_multiple_stocks_data(list(selected_stocks_v3), '2020-01-01', '2025-04-30').pct_change()

# Get the percentage change (returns) for the selected benchmark tickers over the testing period
bx_r_filt_v2 = get_multiple_stocks_data(list(selected_bx),'2020-01-01', '2025-04-30').pct_change()

# Extract the portfolio weights from the recommendation for rebalancing
#weights_series_go = recommendation['weights']

risk_profiles_weights = {
    "DEFENSIVE":    {"min_weight": 0.00002, "max_weight": 0.19},
    "CONSERVATIVE": {"min_weight": 0.00055, "max_weight": 0.2},
    "CAUTIOUS":     {"min_weight": 0.0046, "max_weight": 0.2},
    "EQUILIBRATED": {"min_weight": 0.051, "max_weight": 0.3},
    "DECIDED":      {"min_weight": 0.052, "max_weight": 0.5},
    "DARING":       {"min_weight": 0.03, "max_weight": 0.5},
    "AGGRESSIVE":   {"min_weight": 0.01, "max_weight": 0.495},
}

prices_df_bench = get_multiple_stocks_data(['^GSPC']+final_portfololio,'2020-01-01', '2025-04-30' )
bench_px_sp = prices_df_bench["^GSPC"]     
bench_px_agg = get_stock_data('AGG', '2020-01-01', '2025-04-30')
