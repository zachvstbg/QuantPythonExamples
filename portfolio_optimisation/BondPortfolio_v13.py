import xlrd
import pandas as pd
import numpy as np
from scipy import optimize

# Input files
loc = 'Bond_Portfolio_Model_v3.xlsx'
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

loc1 = ("C:\\Users\\Neurosis\\PycharmProjects\\QuantPythonExamples\\portfolio_optimisation\\Bond_Portfolio_Model_v2.xlsx")
wb1 = xlrd.open_workbook(loc1)
sheet1 = wb1.sheet_by_index(0)

# Extract the vector/matrix data
df_discount = pd.read_excel(wb, usecols='T:AN', dtype=float, header=None, engine='xlrd', skiprows=2, nrows=1)
df_liability_cashflows = pd.read_excel(wb, usecols='U:AN', dtype=float, header=None, engine='xlrd', skiprows=3, nrows=1)
df_liability_npv = pd.read_excel(wb, usecols='T:AM', dtype=float, header=None, engine='xlrd', skiprows=4, nrows=1)

df_asset_cfs = pd.read_excel(wb, usecols='U:AN', dtype=float, header=None, engine='xlrd', skiprows=28, nrows=200)
df_asset_type = pd.read_excel(wb, usecols='I:I', dtype=str, header=None, engine='xlrd', skiprows=28, nrows=200)
df_asset_rating = pd.read_excel(wb, usecols='L:L', dtype=str, header=None, engine='xlrd', skiprows=28, nrows=200)
df_asset_capital = pd.read_excel(wb, usecols='O:O', dtype=float, header=None, engine='xlrd', skiprows=28, nrows=200)
df_asset_MV = pd.read_excel(wb, usecols='Q:Q', dtype=float, header=None, engine='xlrd', skiprows=28, nrows=200)

# Proxy Vol Constraint
# correlation matrix
df02 = pd.read_excel(wb1, usecols="E:F", dtype=float, engine='xlrd', skiprows=0, nrows=2)
corr_matrix = df02.to_numpy()
# shocks = vectors 200x1
pc1_v = [round(sheet1.cell_value(i, 41)) for i in range(23, 223)]
pc2_v = [round(sheet1.cell_value(i, 42)) for i in range(23, 223)]
pc_matrix = np.array([pc1_v, pc2_v])  # 2 x 200

# Set the parameters for constraints (from spreadsheet)
param_gap = [-0.024, 0.024]
param_sov = [0.10, 0.30]
param_corp = [0.00, 0.60]
param_illiq = [0.10, 0.30]
param_bbb = [0.00, 0.40]
param_bb = [0.00, 0.10]
param_loading = 1.50

# Numpy arrays
size_x = 200
np_mv = df_asset_MV.to_numpy().flatten()  # 200 x 1
np_cap_pct = df_asset_capital.to_numpy().flatten()  # 200 x 1
np_asset_cfs = df_asset_cfs.to_numpy()
np_dfs = df_discount.to_numpy().flatten()
np_liab_cfs = df_liability_cashflows.to_numpy().flatten()
np_liab_mv = df_liability_npv.to_numpy().flatten()
#print(np_liab_mv[0])

# The problem via scipy.optimize.minimize
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

# Combining the matrices and bounds for inequality constraints
# This is 200 x 40 matrix (first 20 columns are for c@CFs <= ub, second 20 columns are for c@-Cfs <= -lb
A_ub = np.concatenate((mtrx_asset_cf_accum, mtrx_asset_cf_accum * -1), axis=1)
b_ub = np.concatenate((vec_cum_asset_cfs_ub, vec_cum_asset_cfs_lb * -1))
#x_bounds = [0.00, None] * 200

#initial guess
x_0l = [round(sheet1.cell_value(i, 13), 2) for i in range(23, 223)]
x_0 = np.array(x_0l)

# The actual optimization
fun = lambda x: x @ vec_cost

constraints = [{'type': 'ineq', 'fun': lambda x: b_ub - (x @ A_ub)}]
               #{'type': 'ineq', 'fun': lambda x: (0.061 * np_liab_mv[0])**2 - (np.transpose(pc_matrix @ x) @ (corr_matrix @ (pc_matrix @ x)))}]

options = {'ftol': '1e-8'}
result = optimize.minimize(fun=fun, x0=x_0, constraints=constraints, method='SLSQP', options=options)        # , bounds=x_bounds

# Output the results
x = result.x
pd.DataFrame(x).to_csv('Output_13_03.csv')

# Output some results
print('======== Solver diagnostics ===================================================================================')
print('Message: ', result.message)
print('Success: ', result.success, ', Status = ', result.status)
print('======== Solution =============================================================================================')
print('Asset NPV = ', '{:,.0f}'.format(x @ np_mv))
print('Capital = ', '{:,.0f}'.format(x @ (np_mv * np_cap_pct)))
print('Asset + Loaded Capital = ', '{:,.0f}'.format(result.fun))
print('Surplus = ', '{:,.0f}'.format(np_liab_mv[0] - result.fun))

