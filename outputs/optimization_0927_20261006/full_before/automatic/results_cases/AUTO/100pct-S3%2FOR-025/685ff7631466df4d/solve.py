import gurobipy as gp
from gurobipy import GRB
facilities = ['S1', 'S2']
supermarkets = ['C1', 'C2']
demand = {'C1': 144, 'C2': 216}
fixed_cost = {'S1': 105.97, 'S2': 85.31}
cost = {'S1': {'C1': 2358.39, 'C2': 1492.08}, 'S2': {'C1': 0.07, 'C2': 52.32}}
M = sum(demand.values())
for i in facilities:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for facility {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for facility {i}')
    for j in supermarkets:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for facility {i}, supermarket {j}')
for j in supermarkets:
    if j not in demand:
        raise ValueError(f'Missing demand for supermarket {j}')
m = gp.Model('FLP')
x = m.addVars(facilities, supermarkets, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(facilities, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in facilities for j in supermarkets)) + gp.quicksum((fixed_cost[i] * y[i] for i in facilities)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in facilities)) == demand[j] for j in supermarkets), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in supermarkets)) <= M * y[i] for i in facilities), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')