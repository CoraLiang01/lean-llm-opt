import gurobipy as gp
from gurobipy import GRB
facilities = ['S1', 'S2']
supermarkets = ['C1', 'C2']
demand = {'C1': 144, 'C2': 216}
fixed_cost = {'S1': 105.97, 'S2': 85.31}
transport_cost = {'S1': {'C1': 2358.39, 'C2': 1492.08}, 'S2': {'C1': 0.07, 'C2': 52.32}}
for i in facilities:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for facility {i}')
    if i not in transport_cost:
        raise ValueError(f'Missing transport cost row for facility {i}')
    for j in supermarkets:
        if j not in transport_cost[i]:
            raise ValueError(f'Missing transport cost for facility {i}, supermarket {j}')
for j in supermarkets:
    if j not in demand:
        raise ValueError(f'Missing demand for supermarket {j}')
m = gp.Model('Facility_Location')
x = m.addVars(facilities, supermarkets, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(facilities, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((transport_cost[i][j] * x[i, j] for i in facilities for j in supermarkets)) + gp.quicksum((fixed_cost[i] * y[i] for i in facilities)), GRB.MINIMIZE)
for j in supermarkets:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    for j in supermarkets:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')