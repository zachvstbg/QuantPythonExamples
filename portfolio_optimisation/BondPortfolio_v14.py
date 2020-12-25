import xlrd
import pandas as pd
import numpy as np
from scipy import optimize

# Input files
loc = 'Bond_Portfolio_Model_v5.xlsx'
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

# Extract the vector/matrix data
df_discount = pd.read_excel(wb, usecols='T:AN', dtype=float, header=None, engine='xlrd', skiprows=2, nrows=1)
df_liability_cashflows = pd.read_excel(wb, usecols='U:AN', dtype=float, header=None, engine='xlrd', skiprows=3, nrows=1)
df_liability_npv = pd.read_excel(wb, usecols='T:AM', dtype=float, header=None, engine='xlrd', skiprows=4, nrows=1)

df_asset_cfs = pd.read_excel(wb, usecols='U:AN', dtype=float, header=None, engine='xlrd', skiprows=38, nrows=200)
df_asset_type = pd.read_excel(wb, usecols='I:I', dtype=str, header=None, engine='xlrd', skiprows=38, nrows=200)
df_asset_rating = pd.read_excel(wb, usecols='L:L', dtype=str, header=None, engine='xlrd', skiprows=38, nrows=200)
df_asset_capital = pd.read_excel(wb, usecols='O:O', dtype=float, header=None, engine='xlrd', skiprows=38, nrows=200)
df_asset_MV = pd.read_excel(wb, usecols='Q:Q', dtype=float, header=None, engine='xlrd', skiprows=38, nrows=200)

# Extract extra data - NPV of assets and PC components
df_asset_NPV = pd.read_excel(wb, usecols='BN:CH', dtype=float, header=None, engine='xlrd', skiprows=38, nrows=200)
df_asset_pc1 = pd.read_excel(wb, usecols='CZ:CZ', dtype=float, header=None, engine='xlrd', skiprows=38, nrows=200)
df_asset_pc2 = pd.read_excel(wb, usecols='DA:DA', dtype=float, header=None, engine='xlrd', skiprows=38, nrows=200)
df_corr = pd.read_excel(wb, usecols='E:F', dtype=float, header=None, engine='xlrd', skiprows=1, nrows=2)

# Set the parameters for constraints (from spreadsheet)
param_gap = [-0.02, 0.02]
param_sov = [0.10, 0.30]
param_corp = [0.10, 0.60]
param_illiq = [0.20, 0.40]
param_bbb = [0.00, 0.40]
param_nig = [0.00, 0.10]
param_loading = 1.50

# Convert to numpy arrays (flatten to arrays)
size_x = 200
np_asset_mv = df_asset_NPV.to_numpy()  # 200 x 20
np_cap_pct = df_asset_capital.to_numpy().flatten()  # 200 x 1
np_asset_cfs = df_asset_cfs.to_numpy()
np_dfs = df_discount.to_numpy().flatten()
np_liab_cfs = df_liability_cashflows.to_numpy().flatten()
np_liab_mv = df_liability_npv.to_numpy().flatten()

# The problem via scipy.optimize.minimize
# ======================================================================================================================
# We want to minimize c@X with A_ub@X <=b_ub, A_eq@X = b_eq, and lb <= X <= ub
# Actual function is maximize -> liab_npv - x * mv - x * mv * capital% * loading
# So we can just minimize : x *( mv + mv * capital% * loading)
# So the vector c can be specified as : c = mv*(1 + capital% * loading)
# i.e we buy 100 MV but we need to have 100+Y (if there is capital requirement associated with it)
vec_cost = np_asset_mv[:, 0] * (1 + np_cap_pct * param_loading)  # Size is (200,)

# Prepare the cumulative cashflows - liabilities.
num_cfs = np_liab_cfs.shape[0]
vec_cum_liab = np_liab_cfs
for i in range(1, num_cfs):
    vec_cum_liab[i] = vec_cum_liab[i - 1] * np_dfs[i] / np_dfs[i + 1] + vec_cum_liab[i]

# Prepare the bounds for cumulative asset cashflows
vec_cum_asset_cfs_lb = vec_cum_liab + param_gap[0] * np_liab_mv
vec_cum_asset_cfs_ub = vec_cum_liab + param_gap[1] * np_liab_mv

# Prepare the accumulation matrix - of accumulation factors
vec_acc_factors = np_dfs[0] / np_dfs[:-1]
mtrx_accumulation = np.zeros([num_cfs, num_cfs])
for i in range(0, num_cfs):
    mtrx_accumulation[i, i:] = vec_acc_factors[0:(num_cfs - i)]

# Prepare the cashflow accumulation matrix - i.e. c@This will return the accumulation of asset cashflows
mtrx_asset_cf_accum = np_asset_cfs @ mtrx_accumulation  # Matrix is 200x20

# Combining the matrices and bounds for inequality constraints
# This is 200 x 40 matrix (first 20 columns are for c@CFs <= ub, second 20 columns are for c@-Cfs <= -lb
A_ub = np.concatenate((mtrx_asset_cf_accum, mtrx_asset_cf_accum * -1), axis=1)
b_ub = np.concatenate((vec_cum_asset_cfs_ub, vec_cum_asset_cfs_lb * -1))

# Set the tenors for time-dependent constraints
tenors = [0, 1, 2, 3, 4]
tenors_bbb = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# For other constraints - on sovereigns
vec_is_sov = np.array((df_asset_type == 'Sovereign') * 1.0).flatten()
mtrx_sov_lb = (np.diag(vec_is_sov) - np.eye(size_x) * param_sov[0])  # Lower for MV_sov >= 10%
mtrx_sov_ub = (np.eye(size_x) * param_sov[1] - np.diag(vec_is_sov))  # Upper for MV_sov <= 30%

# For other constraints - on corporates
vec_is_corp = np.array((df_asset_type == 'Corporate') * 1.0).flatten()
mtrx_corp_lb = (np.diag(vec_is_corp) - np.eye(size_x) * param_corp[0])  # Lower for MV_corp >= 0%
mtrx_corp_ub = (np.eye(size_x) * param_corp[1] - np.diag(vec_is_corp))  # Upper for MV_corp <= 60%

# For other constraints - on illiquids
vec_is_illiq = np.array((df_asset_type == 'Illiquid') * 1.0).flatten()
mtrx_illiq_lb = (np.diag(vec_is_illiq) - np.eye(size_x) * param_illiq[0])  # Lower for MV_corp >= 0%
mtrx_illiq_ub = (np.eye(size_x) * param_illiq[1] - np.diag(vec_is_illiq))  # Upper for MV_corp <= 60%

# For other constraints - on BBB
vec_is_BBB = np.array((df_asset_rating == 'BBB') * 1.0).flatten()
mtrx_BBB_lb = (np.diag(vec_is_BBB) - np.eye(size_x) * param_bbb[0])  # Lower for BBB >= 0%
mtrx_BBB_ub = (np.eye(size_x) * param_bbb[1] - np.diag(vec_is_BBB))  # Upper for BBB <= 40%

# For other constraints - on l
vec_is_BB = np.array((df_asset_rating == 'BB') * 1.0).flatten()
vec_is_B = np.array((df_asset_rating == 'B') * 1.0).flatten()
vec_sub_BB = vec_is_BB + vec_is_B
mtrx_subBBB_lb = (np.diag(vec_sub_BB) - np.eye(size_x) * param_nig[0])  # Upper for subBBB >= 0%
mtrx_subBBB_ub = (np.eye(size_x) * param_nig[1] - np.diag(vec_sub_BB))  # Upper for subBBB <= 10%

# Define target function and constraints
fun = lambda x: x @ vec_cost
options = {'disp': True, 'maxiter': 5000}

# Set the initial guess and bounds for x
x_0 = np.ones([200, 1]).flatten() * 0.0001
x_bounds = [(0.00001, None) for i in range(0, 200)]

# Set the general constraint
constraints = [{'type': 'ineq', 'fun': lambda x: x @ mtrx_asset_cf_accum - vec_cum_asset_cfs_lb},  # Mismatch LB
               {'type': 'ineq', 'fun': lambda x: vec_cum_asset_cfs_ub - x @ mtrx_asset_cf_accum},  # Mismatch UB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_sov_lb @ np_asset_mv[:, tenors])},  # Sov LB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_sov_ub @ np_asset_mv[:, tenors])},  # Sov UB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_corp_lb @ np_asset_mv[:, tenors])},  # Corp LB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_corp_ub @ np_asset_mv[:, tenors])},  # Corp UB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_illiq_lb @ np_asset_mv[:, tenors])},  # Corp LB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_illiq_ub @ np_asset_mv[:, tenors])},  # Corp UB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_BBB_ub @ np_asset_mv[:, tenors_bbb])},  # BBB UB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_subBBB_lb @ np_asset_mv[:, tenors_bbb])},  # subBBB LB
               {'type': 'ineq', 'fun': lambda x: x @ (mtrx_subBBB_ub @ np_asset_mv[:, tenors_bbb])}]  # subBBB UB

# Try 1: COBYLA (doesn't support bounds so need to add an additional constraint)
constraints.append({'type': 'ineq', 'fun': lambda x: x - x_0})
result = optimize.minimize(fun=fun, x0=x_0, constraints=constraints, method='COBYLA',
                           options=options)

# Try 2: TRUST-CONSTR (supports bounds)
# result = optimize.minimize(fun=fun, x0=x_0, constraints=constraints, method='trust-constr',
#                            options=options, bounds=x_bounds)

# Output the results
x = result.x
pd.DataFrame(x).to_csv('Output_05.csv')

a = 1
out_npv_asset = x @ np_asset_mv[:, 0]
out_npv_sovereign = x @ (np_asset_mv[:, 0] @ np.diag(vec_is_sov))
out_npv_corporate = x @ (np_asset_mv[:, 0] @ np.diag(vec_is_corp))
out_npv_illiquid = x @ (np_asset_mv[:, 0] @ np.diag(vec_is_illiq))
out_npv_bbb = x @ (np_asset_mv[:, 0] @ np.diag(vec_is_BBB))
out_npv_bb = x @ (np_asset_mv[:, 0] @ np.diag(vec_is_BB))
out_npv_b = x @ (np_asset_mv[:, 0] @ np.diag(vec_is_B))

# Output some results
print('======== Solver diagnostics ===================================================================================')
print('Message: ', result.message)
print('Success: ', result.success, ', Status = ', result.status)
print('======== Solution =============================================================================================')
print('Asset NPV = ', '{:,.0f}'.format(out_npv_asset))
print('Capital = ', '{:,.0f}'.format(x @ (np_asset_mv[:, 0] * np_cap_pct)))
print('Asset + Loaded Capital = ', '{:,.0f}'.format(result.fun))
print('Surplus = ', '{:,.0f}'.format(np_liab_mv[0] - result.fun))
print('======== Breakdown=============================================================================================')
print('Sovereign = ', '{:,.0f}'.format(out_npv_sovereign), '({:,.2%})'.format(out_npv_sovereign / out_npv_asset))
print('Corporate = ', '{:,.0f}'.format(out_npv_corporate), '({:,.2%})'.format(out_npv_corporate / out_npv_asset))
print('Illiquid = ', '{:,.0f}'.format(out_npv_illiquid), '({:,.2%})'.format(out_npv_illiquid / out_npv_asset))
print('Rating BBB = ', '{:,.0f}'.format(out_npv_bbb), '({:,.2%})'.format(out_npv_bbb / out_npv_asset))
print('Rating BB = ', '{:,.0f}'.format(out_npv_bb), '({:,.2%})'.format(out_npv_bb / out_npv_asset))
print('Rating B = ', '{:,.0f}'.format(out_npv_b), '({:,.2%})'.format(out_npv_b / out_npv_asset))
