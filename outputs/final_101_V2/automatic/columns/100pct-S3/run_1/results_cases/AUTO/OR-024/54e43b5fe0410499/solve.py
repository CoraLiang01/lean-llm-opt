import gurobipy as gp
from gurobipy import GRB
facilities = ['S1', 'S2', 'S3']
customers = ['C1', 'C2', 'C3']
demand = {'C1': 1083, 'C2': 776, 'C3': 16214}
fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
cost = {'S1': {'C1': 1506.22, 'C2': 70.9, 'C3': 8.44}, 'S2': {'C1': 1732.65, 'C2': 1780.72, 'C3': 567.44}, 'S3': {'C1': 115.66, 'C2': 100.76, 'C3': 64.68}}
M = 18073
for i in facilities:
    if i not in fixed_cost or i not in cost:
        raise ValueError(f'Missing fixed_cost or cost for facility {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for facility {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
m = gp.Model('Bandcamp_Facility_Location')
x = m.addVars(facilities, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(facilities, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in facilities for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in facilities)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in facilities)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in facilities), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')