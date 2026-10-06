import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    suppliers = ['S1', 'S2']
    supermarkets = ['C1', 'C2']
    demand = {'C1': 144, 'C2': 216}
    fixed_cost = {'S1': 105.97, 'S2': 85.31}
    cost = {'S1': {'C1': 2358.39, 'C2': 1492.08}, 'S2': {'C1': 0.07, 'C2': 52.32}}
    M = 360
    for i in suppliers:
        if i not in fixed_cost or i not in cost:
            raise ValueError(f'Missing fixed or cost data for supplier {i}')
        for j in supermarkets:
            if j not in cost[i]:
                raise ValueError(f'Missing cost data for supplier {i}, supermarket {j}')
    for j in supermarkets:
        if j not in demand:
            raise ValueError(f'Missing demand data for supermarket {j}')
    m = gp.Model('Facility_Location')
    x = m.addVars(suppliers, supermarkets, lb=0, vtype=GRB.CONTINUOUS, name='x')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='y')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in supermarkets)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in supermarkets), name='demand')
    m.addConstrs((gp.quicksum((x[i, j] for j in supermarkets)) <= M * y[i] for i in suppliers), name='activation')
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