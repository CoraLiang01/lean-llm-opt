import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    demand = {'C1': 216, 'C2': 216, 'C3': 216, 'C4': 144, 'C5': 144, 'C6': 144}
    fixed_cost = {'S1': 98.88, 'S2': 99.73, 'S3': 94.01, 'S4': 93.77, 'S5': 107.59, 'S6': 112.65}
    cost = {'S1': {'C1': 0.08, 'C2': 52.33, 'C3': 73.57, 'C4': 1237.33, 'C5': 0.07, 'C6': 112.16}, 'S2': {'C1': 46.02, 'C2': 175.23, 'C3': 2026.83, 'C4': 299.89, 'C5': 966.53, 'C6': 1590.42}, 'S3': {'C1': 1031.74, 'C2': 78.13, 'C3': 99.02, 'C4': 277.07, 'C5': 884.45, 'C6': 1800.86}, 'S4': {'C1': 868.75, 'C2': 94.2, 'C3': 1776.34, 'C4': 285.48, 'C5': 868.85, 'C6': 86.55}, 'S5': {'C1': 1577, 'C2': 760.15, 'C3': 2090.19, 'C4': 43.2, 'C5': 1577.12, 'C6': 1095.17}, 'S6': {'C1': 49.14, 'C2': 4.33, 'C3': 2079.57, 'C4': 277.04, 'C5': 1032.01, 'C6': 1543.49}}
    M = 1080
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('Adidas_Facility_Location')
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()