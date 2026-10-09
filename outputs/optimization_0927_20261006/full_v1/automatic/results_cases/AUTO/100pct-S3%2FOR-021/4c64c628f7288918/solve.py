import gurobipy as gp
from gurobipy import GRB
plants = ['S1', 'S2', 'S3', 'S4']
outlets = ['C1', 'C2', 'C3', 'C4']
demand = {'C1': 94, 'C2': 39, 'C3': 65, 'C4': 435}
supply_capacity = {'S1': 2531, 'S2': 20, 'S3': 210, 'S4': 241}
cost = {'S1': {'C1': 543.756480860856, 'C2': 23.685276141764653, 'C3': 23.676386730773032, 'C4': 447.75143678673766}, 'S2': {'C1': 883.9151090405642, 'C2': 0.04977684765576961, 'C3': 0.0350986687216299, 'C4': 44.45588531711622}, 'S3': {'C1': 537.3456896658107, 'C2': 23.769274659075112, 'C3': 498.95659249465467, 'C4': 440.60737890439776}, 'S4': {'C1': 1791.493192397229, 'C2': 68.21633865655126, 'C3': 1432.4837339656747, 'C4': 1527.7635425462734}}
for i in plants:
    if i not in cost or i not in supply_capacity:
        raise ValueError(f'Missing cost or supply_capacity for plant {i}')
    for j in outlets:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for plant {i}, outlet {j}')
for j in outlets:
    if j not in demand:
        raise ValueError(f'Missing demand for outlet {j}')

def build_and_solve():
    model = gp.Model('BrewCo_Transportation')
    model.Params.MIPGap = 0.0001
    x_vars = model.addVars(plants, outlets, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in plants for j in outlets)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum((x_vars[i, j] for i in plants)) >= demand[j] for j in outlets), name='')
    model.addConstrs((gp.quicksum((x_vars[i, j] for j in outlets)) <= supply_capacity[i] for i in plants), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = build_and_solve()