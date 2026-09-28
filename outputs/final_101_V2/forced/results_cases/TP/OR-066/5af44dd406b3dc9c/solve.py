import gurobipy as gp
from gurobipy import GRB
suppliers = ['S1', 'S2']
customers = ['C1', 'C2']
fixed_cost = {'S1': 105.97, 'S2': 85.31}
transport_cost = {'S1': {'C1': 2358.39, 'C2': 1492.08}, 'S2': {'C1': 0.07, 'C2': 52.32}}
demand = {'C1': 144, 'C2': 216}
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    if i not in transport_cost:
        raise ValueError(f'Missing transport cost row for supplier {i}')
    for j in customers:
        if j not in transport_cost[i]:
            raise ValueError(f'Missing transport cost for supplier {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
m = gp.Model('Supplier_Activation')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
for j in customers:
    m.addConstr(x['S1', j] + x['S2', j] == demand[j], name=f'demand_{j}')
total_demand = sum((demand[j] for j in customers))
for i in suppliers:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= total_demand * y[i], name=f'link_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')