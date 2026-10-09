from gurobipy import Model, GRB, quicksum
suppliers = ['S1', 'S2']
supermarkets = ['C1', 'C2']
fixed_cost = {'S1': 105.97, 'S2': 85.31}
demand = {'C1': 144, 'C2': 216}
transport_cost = {('S1', 'C1'): 2358.39, ('S1', 'C2'): 1492.08, ('S2', 'C1'): 0.07, ('S2', 'C2'): 52.32}
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
for j in supermarkets:
    if j not in demand:
        raise ValueError(f'Missing demand for supermarket {j}')
for i in suppliers:
    for j in supermarkets:
        if (i, j) not in transport_cost:
            raise ValueError(f'Missing transport cost for supplier {i} to supermarket {j}')
m = Model()
m.Params.MIPGap = 0.0001
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
x_vars = m.addVars(suppliers, supermarkets, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)) + quicksum((transport_cost[i, j] * x_vars[i, j] for i in suppliers for j in supermarkets)), GRB.MINIMIZE)
for j in supermarkets:
    m.addConstr(quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(x_vars[i, 'C1'] <= demand['C1'] * y_vars[i], name=f'link_{i}_C1')
    m.addConstr(x_vars[i, 'C2'] <= demand['C2'] * y_vars[i], name=f'link_{i}_C2')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')