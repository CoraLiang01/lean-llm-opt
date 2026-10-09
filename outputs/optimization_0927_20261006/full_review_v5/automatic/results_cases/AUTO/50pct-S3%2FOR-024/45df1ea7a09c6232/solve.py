import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3']
musicians = ['C1', 'C2', 'C3']
demand = {'C1': 1083, 'C2': 776, 'C3': 16214}
fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
cost = {'S1': {'C1': 1506.22, 'C2': 70.9, 'C3': 8.44}, 'S2': {'C1': 1732.65, 'C2': 1780.72, 'C3': 567.44}, 'S3': {'C1': 115.66, 'C2': 100.76, 'C3': 64.68}}
M = sum(demand.values())
for i in warehouses:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed_cost for warehouse {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for warehouse {i}')
    for j in musicians:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for warehouse {i}, musician {j}')
for j in musicians:
    if j not in demand:
        raise ValueError(f'Missing demand for musician {j}')
m = gp.Model('Bandcamp_Facility_Location')
x_vars = m.addVars(warehouses, musicians, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in musicians)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j] for j in musicians), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in musicians)) <= M * y_vars[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')