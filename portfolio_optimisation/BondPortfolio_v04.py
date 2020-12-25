# import pulp
import xlrd
import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
import pandas as pd
from scipy import optimize

loc = ("Bond_Portfolio_Model_v2.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

# c 200 x 1     mv vector copied from excel sheet
bonds_mv = [round(sheet.cell_value(i, 16)) for i in range(23, 223)]
capital_coefficient = [round(1 + 1.5 * sheet.cell_value(i, 14), 5) for i in range(23, 223)]
cost_vector = np.array(capital_coefficient) * np.array(bonds_mv)
# print(cost_vector)
# print(len(cost_vector))


# Funding gap constraint
liabilities_NPV = 762827554
liabilities_per_year = [round(sheet.cell_value(3, i)) for i in range(20, 40)]           # 1 x 20
funding_gap_max = [round(sheet.cell_value(20, i)) for i in range(20, 40)]               # 1 x 20
funding_gap_upper = [liabilities_per_year[i] + funding_gap_max[i] for i in range(20)]   # 1 x 20
funding_gap_lower = [(liabilities_per_year[i] - funding_gap_max[i]) for i in range(20)]   # 1 x 20
# actually payments bounds  1 x 40:
funding_gap_bounds = funding_gap_upper + funding_gap_lower
#print('Payments bounds: ', funding_gap_bounds, 'length: ', len(funding_gap_bounds))


# extract the payments matrix from Excel (I name it main_matrix)
df01 = pd.read_excel(wb, usecols="U:AN", dtype=float, engine='xlrd', skiprows=22)
main_matrix = df01.to_numpy()           # payments matrix 200 x 20


# constraints by type of bonds / tenor
# upper limits
sov_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Sovereign' else 0 for i in range(23, 223)]
corp_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Corporate' else 0 for i in range(23, 223)]
illiquid_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Illiquid' else 0 for i in range(23, 223)]
bbb_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 11) == 'BBB' else 0 for i in range(23, 223)]
below_bbb_mv = [int(sheet.cell_value(i, 16)) if (sheet.cell_value(i, 11) == 'BB' or sheet.cell_value(i, 11) == 'B') else 0 for i in range(23, 223)]
# lower limits
sov_mv_l = np.negative(sov_mv)
illiquid_mv_l = np.negative(illiquid_mv)


# constraints_matrix = np.transpose([sov_mv, sov_mv_l, corp_mv, illiquid_mv, illiquid_mv_l,  bbb_mv, below_bbb_mv])  # sov_mv, sov_mv_l, corp_mv, illiquid_mv, illiquid_mv_l,  bbb_mv, below_bbb_mv
# #print(constraints_matrix)

# # define the A_ub matrix
# #    rows:  47                  20              20                7
# opt_matrix = np.concatenate((main_matrix, funding_gap_matrix, constraints_matrix), axis=1)    # 200 x 47
# #opt_matrix = np.concatenate((main_matrix, funding_gap_matrix), axis=1)

# # prepare arguments
# bonds_mv_T = np.transpose(bonds_mv)               # 200 x 1
# opt_matrix_T = np.transpose(opt_matrix)           # 47 x 200
# bounds_vector_T = np.transpose(bounds_vector)     # 47 x 1
# #bounds_vector_T = np.transpose(funding_gap_bounds)


#Model solver:
model = LpProblem(name = "BondPortfolio", sense = LpMaximize)
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(200)}

# objective:
model += liabilities_NPV - lpSum(cost_vector[i] * x[i] for i in range(200))

# funding gap constraints
for i in range(20):
    model += lpSum(x[j] * main_matrix[j, i] for j in range(200)) <= funding_gap_upper[i]
    model += lpSum(x[j] * main_matrix[j, i] for j in range(200)) >= funding_gap_lower[i]

status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in x.values():
    print(f"{var.value():.2f}")

result = [var.value() for var in x.values()]
pd.DataFrame(result).to_csv('Output_04.csv')