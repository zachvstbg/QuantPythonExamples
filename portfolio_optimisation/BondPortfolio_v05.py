import xlrd
import pandas as pd
import numpy as np
from scipy import optimize

# Input files
loc = 'Bond_Portfolio_Model_v3.xlsx'
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

# Extract the vector/matrix data
df_discount = pd.read_excel(wb, usecols='T:AN', dtype=float, header=None, engine='xlrd', skiprows=2, nrows=1)
df_liability_cashflows = pd.read_excel(wb, usecols='U:AN', dtype=float, header=None, engine='xlrd', skiprows=3, nrows=1)
df_liability_npv = pd.read_excel(wb, usecols='T:AM', dtype=float, header=None, engine='xlrd', skiprows=4, nrows=1)

df_asset_cfs = pd.read_excel(wb, usecols='U:AN', dtype=float, header=None, engine='xlrd', skiprows=28, nrows=200)
df_asset_type = pd.read_excel(wb, usecols='I:I', dtype=str, header=None, engine='xlrd', skiprows=28, nrows=200)
df_asset_rating = pd.read_excel(wb, usecols='L:L', dtype=str, header=None, engine='xlrd', skiprows=28, nrows=200)
df_asset_capital = pd.read_excel(wb, usecols='O:O', dtype=float, header=None, engine='xlrd', skiprows=28, nrows=200)
df_asset_MV = pd.read_excel(wb, usecols='Q:Q', dtype=float, header=None, engine='xlrd', skiprows=28, nrows=200)

# Set the parameters for constraints (from spreadsheet)
param_gap = [-0.015, 0.015]
param_sov = [0.10, 0.30]
param_corp = [0.00, 0.60]
param_illiq = [0.10, 0.30]
param_bbb = [0.00, 0.40]
param_nig = [0.05, 0.10]
param_loading = 1.50

# Numpy arrays
size_x = 200
np_mv = df_asset_MV.to_numpy().flatten()  # 200 x 1
np_cap_pct = df_asset_capital.to_numpy().flatten()  # 200 x 1
np_asset_cfs = df_asset_cfs.to_numpy()
np_dfs = df_discount.to_numpy().flatten()
np_liab_cfs = df_liability_cashflows.to_numpy().flatten()
np_liab_mv = df_liability_npv.to_numpy().flatten()

# The problem via scipy.optimize.linprog
# ======================================================================================================================
# We want to minimize c@X with A_ub@X <=b_ub, A_eq@X = b_eq, and lb <= X <= ub
# Actual function is maximize -> liab_npv - x * mv - x * mv * capital% * loading
# So we can just minimize : x *( mv + mv * capital% * loading)
# So the vector c can be specified as : c = mv*(1 + capital% * loading)
# i.e we buy 100 MV but we need to have 100+Y (if there is capital requirement associated with it)

vec_cost = np_mv * (1 + np_cap_pct * param_loading)  # Size is (200,)

# Prepare the cumulative cashflows - liabilities.
# This is calculating row 22 in the Excel model
num_cfs = np_liab_cfs.shape[0]
vec_cum_liab = np_liab_cfs
for i in range(1, num_cfs):
    vec_cum_liab[i] = vec_cum_liab[i - 1] * np_dfs[i] / np_dfs[i + 1] + vec_cum_liab[i]

# Prepare the bounds for cumulative asset cashflows
vec_cum_asset_cfs_lb = vec_cum_liab + param_gap[0] * np_liab_mv  # For row 24 in Excel model
vec_cum_asset_cfs_ub = vec_cum_liab + param_gap[1] * np_liab_mv  # For row 25 in Excel model

# Prepare the accumulation matrix - of accumulation factors
vec_acc_factors = np_dfs[0] / np_dfs[:-1]
mtrx_accumulation = np.zeros([num_cfs, num_cfs])
for i in range(0, num_cfs):
    mtrx_accumulation[i, i:] = vec_acc_factors[0:(num_cfs - i)]

# Prepare the cashflow accumulation matrix - i.e. c@This will return the accumulation of asset cashflows
mtrx_asset_cf_accum = np_asset_cfs @ mtrx_accumulation  # Matrix is 200x20

# True/False vectors for identifying the asset-related constraints
vec_is_sov = np.array((df_asset_type == 'Sovereign') * 1.0).flatten()
vec_is_corp = np.array((df_asset_type == 'Corporate') * 1.0).flatten()
vec_is_illiq = np.array((df_asset_type == 'Illiquid') * 1.0).flatten()
vec_is_BBB = np.array((df_asset_rating == 'BBB') * 1.0).flatten()
vec_is_BB = np.array((df_asset_rating == 'BB') * 1.0).flatten()
vec_is_B = np.array((df_asset_rating == 'B') * 1.0).flatten()

# Define the matrix and constraints
mtrx_sov_lb = np_mv @ (np.eye(size_x) * param_sov[0] - np.diag(vec_is_sov))  # Lower for MV_sov >= 10%
mtrx_sov_ub = np_mv @ (np.diag(vec_is_sov) - np.eye(size_x) * param_sov[1])  # Upper for MV_sov <= 30%
mtrx_corp_lb = np_mv @ (np.eye(size_x) * param_corp[0] - np.diag(vec_is_corp))  # Lower for MV_corp >= 0%
mtrx_corp_ub = np_mv @ (np.diag(vec_is_corp) - np.eye(size_x) * param_corp[1])  # Upper for MV_corp <= 60%
mtrx_illiq_lb = np_mv @ (np.eye(size_x) * param_illiq[0] - np.diag(vec_is_illiq))  # Lower for MV_illiq >= 10%
mtrx_illiq_ub = np_mv @ (np.diag(vec_is_illiq) - np.eye(size_x) * param_illiq[1])  # Upper for MV_illiq <= 30%
mtrx_BBB_lb = np_mv @ (np.eye(size_x) * param_bbb[0] - np.diag(vec_is_BBB))  # Lower for MV_BBB >= 0%
mtrx_BBB_ub = np_mv @ (np.diag(vec_is_BBB) - np.eye(size_x) * param_bbb[1])  # Upper for MV_BBB <= 40%
mtrx_NIG_lb = np_mv @ (np.eye(size_x) * param_nig[0] - np.diag(vec_is_BB + vec_is_B))  # Lower for MV_BB >= 0%
mtrx_NIG_ub = np_mv @ (np.diag(vec_is_BB + vec_is_B) - np.eye(size_x) * param_nig[1])  # Upper for MV_BB <= 10%

# Stack the vectors
constraints = np.vstack([mtrx_sov_lb, mtrx_sov_ub, mtrx_corp_lb, mtrx_corp_ub, mtrx_illiq_lb, mtrx_illiq_ub,
                         mtrx_BBB_lb, mtrx_BBB_ub, mtrx_NIG_lb, mtrx_NIG_ub])

# Combining the matrices and bounds for inequality constraints
# This is 200 x 40 matrix (first 20 columns are for c@CFs <= ub, second 20 columns are for c@-Cfs <= -lb
A_ub = np.concatenate((mtrx_asset_cf_accum, mtrx_asset_cf_accum * -1, constraints.transpose()), axis=1)
b_ub = np.concatenate((vec_cum_asset_cfs_ub, vec_cum_asset_cfs_lb * -1, np.zeros(10)))
x_bounds = (0.0000, None)

# The actual optimization
options = {'presolve': True, 'autoscale': True, 'tol': 1e-10}
result = optimize.linprog(c=vec_cost, A_ub=A_ub.transpose(), b_ub=b_ub.transpose(), options=options,
                          bounds=x_bounds, method='interior-point')

# Output the results
x = result.x
pd.DataFrame(x).to_csv('Output_05.csv')

out_npv_asset = x @ np_mv
out_npv_sovereign = x@(np_mv@np.diag(vec_is_sov))
out_npv_corporate = x@(np_mv@np.diag(vec_is_corp))
out_npv_illiquid = x@(np_mv@np.diag(vec_is_illiq))
out_npv_bbb = x@(np_mv@np.diag(vec_is_BBB))
out_npv_bb = x@(np_mv@np.diag(vec_is_BB))
out_npv_b = x@(np_mv@np.diag(vec_is_B))

# Output some results
print('======== Solver diagnostics ===================================================================================')
print('Message: ', result.message)
print('Success: ', result.success, ', Status = ', result.status)
print('======== Solution =============================================================================================')
print('Asset NPV = ', '{:,.0f}'.format(out_npv_asset))
print('Capital = ', '{:,.0f}'.format(x @ (np_mv * np_cap_pct)))
print('Asset + Loaded Capital = ', '{:,.0f}'.format(result.fun))
print('Surplus = ', '{:,.0f}'.format(np_liab_mv[0] - result.fun))
print('======== Breakdown=============================================================================================')
print('Sovereign = ', '{:,.0f}'.format(out_npv_sovereign), '({:,.2%})'.format(out_npv_sovereign/out_npv_asset))
print('Corporate = ', '{:,.0f}'.format(out_npv_corporate), '({:,.2%})'.format(out_npv_corporate/out_npv_asset))
print('Illiquid = ', '{:,.0f}'.format(out_npv_illiquid), '({:,.2%})'.format(out_npv_illiquid/out_npv_asset))
print('Rating BBB = ', '{:,.0f}'.format(out_npv_bbb), '({:,.2%})'.format(out_npv_bbb/out_npv_asset))
print('Rating BB = ', '{:,.0f}'.format(out_npv_bb), '({:,.2%})'.format(out_npv_bb/out_npv_asset))
print('Rating B = ', '{:,.0f}'.format(out_npv_b), '({:,.2%})'.format(out_npv_b/out_npv_asset))