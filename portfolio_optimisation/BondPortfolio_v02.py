# import pulp
import xlrd
import numpy as np
# from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
import pandas as pd
from scipy import optimize

loc = ("Bond_Portfolio_Model_v2.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

# c 1 x 200     mv vector copied from excel sheet
bonds_mv = [round(sheet.cell_value(i, 16)) for i in range(23, 223)]
capital_coefficient = [round(1 + 1.5 * sheet.cell_value(i, 14), 5) for i in range(23, 223)]
cost_vector = np.array(capital_coefficient) * np.array(bonds_mv)
# print(cost_vector)
# print(bonds_mv)


# Constraints vector b_up parts:
# Funding gap constraint
liabilities_NPV = 762827554
liabilities_per_year = [round(sheet.cell_value(3, i)) for i in range(20, 40)]  # 1 x 20
funding_gap_max = [round(sheet.cell_value(20, i)) for i in range(20, 40)]  # 1 x 20
funding_gap_upper = [liabilities_per_year[i] + funding_gap_max[i] for i in range(20)]  # 1 x 20
funding_gap_lower = [(liabilities_per_year[i] - funding_gap_max[i]) * -1 for i in range(20)]  # 1 x 20, negative sign
# actually payments bounds:
funding_gap_bounds = funding_gap_upper + funding_gap_lower
# print('Payments bounds: ', funding_gap_bounds, 'length: ', len(funding_gap_bounds))

asset_NPV = 624000000  # asset_NPV = 718655828 we use liabilities_NPV for now (broader bounds intervals  )
# constraints by bond type / tenor
sov_upper = 0.3 * asset_NPV
sov_lower = -0.1 * asset_NPV
corp_upper = 0.6 * asset_NPV
illiquid_upper = 0.3 * asset_NPV
illiquid_lower = -0.1 * asset_NPV
bbb_upper = 0.4 * asset_NPV
below_bbb_upper = 0.1 * asset_NPV
constraints = [sov_upper, sov_lower, corp_upper, illiquid_upper, illiquid_lower, bbb_upper,
               below_bbb_upper]  # sov_upper, sov_lower, corp_upper, illiquid_upper, illiquid_lower, bbb_upper, below_bbb_upper

# define the b_ub vector to be passed as argument to linprog
#   length: 47                         40                7
bounds_vector = np.concatenate((funding_gap_bounds, constraints))

# Matrix A_ub parts:
# extract the payments matrix from Excel (I name it main_matrix)
df01 = pd.read_excel(wb, usecols="U:AN", dtype=float, engine='xlrd', skiprows=22)
main_matrix = df01.to_numpy()  # payments matrix 200 x 20

# funding gap constraints - main matrix sufficient for max constraints and we need negative matrix for mins
funding_gap_matrix = np.negative(main_matrix)  # 200 x 20

# constraints by type of bonds / tenor
# upper limits
sov_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Sovereign' else 0 for i in range(23, 223)]
corp_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Corporate' else 0 for i in range(23, 223)]
illiquid_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Illiquid' else 0 for i in range(23, 223)]
bbb_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 11) == 'BBB' else 0 for i in range(23, 223)]
below_bbb_mv = [
    int(sheet.cell_value(i, 16)) if (sheet.cell_value(i, 11) == 'BB' or sheet.cell_value(i, 11) == 'B') else 0 for i in
    range(23, 223)]
# lower limits
sov_mv_l = np.negative(sov_mv)
illiquid_mv_l = np.negative(illiquid_mv)
# print(sov_lower)
# print(sov_mv_l)

constraints_matrix = np.transpose([sov_mv, sov_mv_l, corp_mv, illiquid_mv, illiquid_mv_l, bbb_mv,
                                   below_bbb_mv])  # sov_mv, sov_mv_l, corp_mv, illiquid_mv, illiquid_mv_l,  bbb_mv, below_bbb_mv
# print(constraints_matrix)

# define the A_ub matrix to be passed as argument to linprog
#    rows:  47                  20              20                7
opt_matrix = np.concatenate((main_matrix, funding_gap_matrix, constraints_matrix), axis=1)  # 200 x 47
# opt_matrix = np.concatenate((main_matrix, funding_gap_matrix), axis=1)

# prepare arguments for linprog
# bonds_mv_T = np.transpose(bonds_mv)               # 200 x 1
cost_vector_T = np.transpose(cost_vector)
opt_matrix_T = np.transpose(opt_matrix)  # 47 x 200
bounds_vector_T = np.transpose(bounds_vector)  # 47 x 1
# bounds_vector_T = np.transpose(funding_gap_bounds)

# 1. result = optimize.linprog(c=bonds_mv_T, A_ub=opt_matrix_T, b_ub=bounds_vector_T, bounds=(0.001, 100), options={'tol': 1e-8} )               #  method='simplex', options={'tol': 1e-6} 'presolve':False, , options={'tol': 1e-10})
# 2. result = optimize.linprog(c=bonds_mv_T, A_ub=opt_matrix_T, b_ub=bounds_vector_T, bounds=(0.01, 60), options={'tol': 1e-6, 'lstsq': True})  #  options={'tol': 1e-6} 'presolve':False, , options={'tol': 1e-10})
# 3. result = optimize.linprog(c=bonds_mv_T, A_ub=opt_matrix_T, b_ub=bounds_vector_T, bounds=(0.0001, 60), options={'lstsq': True})      #  options={'tol': 1e-6} 'presolve':False, , options={'tol': 1e-10})
# 4. result = optimize.linprog(c=bonds_mv_T, A_ub=opt_matrix_T, b_ub=bounds_vector_T, bounds=(0.001, 100), options={'lstsq': True})
# 5. result = optimize.linprog(c=bonds_mv_T, A_ub=opt_matrix_T, b_ub=bounds_vector_T)
result = optimize.linprog(c=cost_vector_T, A_ub=opt_matrix_T, b_ub=bounds_vector_T)
print('Message: ', result.message)
print('Success: ', result.success, ', Status = ', result.status)
print('To maximize(liabilities_NPV - (asset_NPV + loaded capital)) = ', liabilities_NPV - result.fun)
asset_NPV_result = np.dot(result.x, bonds_mv)
print('Asset NPV: ', asset_NPV_result)
print('Loaded capital: ', result.fun - asset_NPV_result)
print('Asset_NPV + Loaded capital: ', result.fun)
print('solution values: ')
for num in result.x:
    print(round(num, 2))

pd.DataFrame(result.x).to_csv('Output_02.csv')
# print('--------------------------------------')
# print(np.matmul(opt_matrix_T, np.transpose(result['x'])))
# print('--------------------------------------')
# print(sov_upper, sov_lower, corp_upper, illiquid_upper, illiquid_lower, bbb_upper, below_bbb_upper)
