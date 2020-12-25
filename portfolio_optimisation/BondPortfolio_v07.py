import xlrd
import numpy as np
from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
import pandas as pd

loc = ("C:\\Users\\Neurosis\\PycharmProjects\\QuantPythonExamples\\portfolio_optimisation\\Bond_Portfolio_Model_v2.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

# c 200 x 1     mv vector copied from excel sheet
bonds_mv = [round(sheet.cell_value(i, 16)) for i in range(23, 223)]
capital_coefficient = [round(1 + 1.5 * sheet.cell_value(i, 14), 5) for i in range(23, 223)]
# The cost vector c can be specified as : c = mv * (1 + capital% * loading)
cost_vector = np.array(capital_coefficient) * np.array(bonds_mv)

# Funding gap constraint
fg = 0.015  # funding gap bound
disc_f = [sheet.cell_value(2, i) for i in range(20, 40)]  # discount factor 1 x 20
liabilities_per_year = [round(sheet.cell_value(3, i)) for i in range(20, 40)]  # 1 x 20
liabilities_NPV = [round(sheet.cell_value(4, i)) for i in range(19, 39)]  # 1 x 20
liabilities_acc = [90000000]  # 1 x 20
for i in range(1, 20):
    acc_value = disc_f[i - 1] / disc_f[i] * liabilities_acc[i - 1] + liabilities_per_year[i]
    liabilities_acc.append(round(acc_value))

# vectors by type of bonds / tenor
sov_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Sovereign' else 0 for i in range(23, 223)]
corp_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Corporate' else 0 for i in range(23, 223)]
illiquid_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 8) == 'Illiquid' else 0 for i in range(23, 223)]
bbb_mv = [int(sheet.cell_value(i, 16)) if sheet.cell_value(i, 11) == 'BBB' else 0 for i in range(23, 223)]
below_bbb_mv = [
    int(sheet.cell_value(i, 16)) if (sheet.cell_value(i, 11) == 'BB' or sheet.cell_value(i, 11) == 'B') else 0 for i in
    range(23, 223)]

# extract the payments matrix from Excel (main_matrix)
df01 = pd.read_excel(wb, usecols="U:AN", dtype=float, engine='xlrd', skiprows=22)
main_matrix = df01.to_numpy()  # payments matrix 200 x 20


def payment(x, yr):
    return lpSum(x[j] * main_matrix[j, yr] for j in range(200))


def acc_payments(x, yr):
    if yr == 0:
        return payment(x, yr)
    else:
        return disc_f[yr - 1] / disc_f[yr] * acc_payments(x, yr - 1) + payment(x, yr)


def fund_gap(x, yr):
    return (acc_payments(x, yr) - liabilities_acc[yr]) / liabilities_NPV[yr]


# Model solver:
model = LpProblem(name="BondPortfolio", sense=LpMaximize)

# variable
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(200)}

# objective asset_NPV + loaded capital:
model += liabilities_NPV[0] - lpSum(cost_vector[i] * x[i] for i in range(200))

# funding gap constraints
for i in range(20):
    model += fund_gap(x, i) <= fg
    model += fund_gap(x, i) >= -fg

# constraints by bond type / tenor
model += lpSum(x[j] * sov_mv[j] for j in range(200)) >= 0.1 * lpSum(x[j] * bonds_mv[j] for j in range(200))
model += lpSum(x[j] * sov_mv[j] for j in range(200)) <= 0.3 * lpSum(x[j] * bonds_mv[j] for j in range(200))
model += lpSum(x[j] * corp_mv[j] for j in range(200)) <= 0.6 * lpSum(x[j] * bonds_mv[j] for j in range(200))
model += lpSum(x[j] * illiquid_mv[j] for j in range(200)) >= 0.1 * lpSum(x[j] * bonds_mv[j] for j in range(200))
model += lpSum(x[j] * illiquid_mv[j] for j in range(200)) <= 0.3 * lpSum(x[j] * bonds_mv[j] for j in range(200))
model += lpSum(x[j] * bbb_mv[j] for j in range(200)) <= 0.4 * lpSum(x[j] * bonds_mv[j] for j in range(200))
model += lpSum(x[j] * below_bbb_mv[j] for j in range(200)) <= 0.1 * lpSum(x[j] * bonds_mv[j] for j in range(200))

# Solve
status = model.solve()

# Print some output
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective (to maximize): {model.objective.value()}")
asset_NPV = liabilities_NPV[0] - model.objective.value()
print('Asset NPV + loaded capital: ', asset_NPV)
for var in x.values():
    print(f"{var.value():.2f}")

