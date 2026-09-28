import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3']
customers = ['C1', 'C2', 'C3']
fixed_costs = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
demands = {'C1': 1083, 'C2': 776, 'C3': 16214}
transportation_costs = {'S1': {'C1': 1506.22, 'C2': 70.9, 'C3': 8.44}, 'S2': {'C1': 1732.65, 'C2': 1780.72, 'C3': 567.44}, 'S3': {'C1': 115.66, 'C2': 100.76, 'C3': 64.68}}
for i in warehouses:
    if i not in fixed_costs:
        raise ValueError(f'Missing fixed cost for warehouse {i}')
    if i not in transportation_costs:
        raise ValueError(f'Missing transportation costs for warehouse {i}')
    for j in customers:
        if j not in transportation_costs[i]:
            raise ValueError(f'Missing transportation cost for warehouse {i}, customer {j}')
for j in customers:
    if j not in demands:
        raise ValueError(f'Missing demand for customer {j}')
m = gp.Model('Bandcamp_Warehouse_Selection')
y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
x = m.addVars(warehouses, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in warehouses)) + gp.quicksum((transportation_costs[i][j] * x[i, j] for i in warehouses for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demands[j] for j in customers), name='')
m.addConstrs((x[i, j] <= demands[j] * y[i] for i in warehouses for j in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')