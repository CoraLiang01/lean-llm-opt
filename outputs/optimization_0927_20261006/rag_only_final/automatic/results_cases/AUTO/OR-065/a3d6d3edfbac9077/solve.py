from gurobipy import Model, GRB, quicksum
warehouses = ['S1', 'S2', 'S3']
customers = ['C1', 'C2', 'C3']
fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
demand = {'C1': 1083, 'C2': 776, 'C3': 16214}
transport_cost = {('S1', 'C1'): 1506.22, ('S1', 'C2'): 70.9, ('S1', 'C3'): 8.44, ('S2', 'C1'): 1732.65, ('S2', 'C2'): 1780.72, ('S2', 'C3'): 567.44, ('S3', 'C1'): 115.66, ('S3', 'C2'): 100.76, ('S3', 'C3'): 64.68}
for i in warehouses:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for warehouse {i}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
for i in warehouses:
    for j in customers:
        if (i, j) not in transport_cost:
            raise ValueError(f'Missing transport cost for ({i},{j})')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(warehouses, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(warehouses, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_cost[i] * y_vars[i] for i in warehouses)) + quicksum((transport_cost[i, j] * x_vars[i, j] for i in warehouses for j in customers)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(quicksum((x_vars[i, j] for i in warehouses)) == demand[j], name=f'demand_{j}')
    for i in warehouses:
        for j in customers:
            m.addConstr(x_vars[i, j] <= demand[j] * y_vars[i], name=f'link_{i}_{j}')
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')