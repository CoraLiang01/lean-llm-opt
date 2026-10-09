import gurobipy as gp
from gurobipy import GRB
suppliers = ['S1', 'S2']
customers = ['C1', 'C2']
demand = {'C1': 144, 'C2': 216}
fixed_cost = {'S1': 105.97, 'S2': 85.31}
cost = {'S1': {'C1': 2358.39, 'C2': 1492.08}, 'S2': {'C1': 0.07, 'C2': 52.32}}
M = 360
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
m = gp.Model('Facility_Location')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((x['S1', j] + x['S2', j] == demand[j] for j in customers), name='')
m.addConstr(x['S1', 'C1'] + x['S1', 'C2'] <= M * y['S1'], name='activate_S1')
m.addConstr(x['S2', 'C1'] + x['S2', 'C2'] <= M * y['S2'], name='activate_S2')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')