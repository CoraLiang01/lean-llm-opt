import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
musicians = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
demand = {'C1': 1083, 'C2': 776, 'C3': 16214, 'C4': 553, 'C5': 17106, 'C6': 594, 'C7': 732}
fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83, 'S4': 98.71, 'S5': 95.73, 'S6': 99.96, 'S7': 98.16}
transportation_cost = {'S1': {'C1': 1506.22, 'C2': 70.9, 'C3': 8.44, 'C4': 260.27, 'C5': 197.47, 'C6': 71.71, 'C7': 61.19}, 'S2': {'C1': 1732.65, 'C2': 1780.72, 'C3': 567.44, 'C4': 448.68, 'C5': 29.0, 'C6': 1484.91, 'C7': 963.92}, 'S3': {'C1': 115.66, 'C2': 100.76, 'C3': 64.68, 'C4': 1324.53, 'C5': 64.99, 'C6': 134.88, 'C7': 2102.83}, 'S4': {'C1': 1254.78, 'C2': 1115.63, 'C3': 52.31, 'C4': 1036.16, 'C5': 892.63, 'C6': 1464.04, 'C7': 1383.41}, 'S5': {'C1': 42.9, 'C2': 891.01, 'C3': 1013.94, 'C4': 1128.72, 'C5': 58.91, 'C6': 42.89, 'C7': 1570.31}, 'S6': {'C1': 0.7, 'C2': 139.46, 'C3': 70.03, 'C4': 79.15, 'C5': 1482.0, 'C6': 0.91, 'C7': 110.46}, 'S7': {'C1': 1732.3, 'C2': 1780.44, 'C3': 486.5, 'C4': 523.74, 'C5': 522.08, 'C6': 82.48, 'C7': 826.41}}
M = sum(demand.values())
for i in warehouses:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for warehouse {i}')
    if i not in transportation_cost:
        raise ValueError(f'Missing transportation cost row for warehouse {i}')
    for j in musicians:
        if j not in transportation_cost[i]:
            raise ValueError(f'Missing transportation cost for warehouse {i}, musician {j}')
for j in musicians:
    if j not in demand:
        raise ValueError(f'Missing demand for musician {j}')
m = gp.Model('Bandcamp_Facility_Location')
x_vars = m.addVars(warehouses, musicians, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((transportation_cost[i][j] * x_vars[i, j] for i in warehouses for j in musicians)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouses)), GRB.MINIMIZE)
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