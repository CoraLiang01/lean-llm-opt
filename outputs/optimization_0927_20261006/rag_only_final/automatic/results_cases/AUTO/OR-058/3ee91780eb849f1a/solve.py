from gurobipy import Model, GRB, quicksum
suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
stores = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
fixed_cost = {'S1': 98.88, 'S2': 99.73, 'S3': 94.01, 'S4': 93.77, 'S5': 107.59, 'S6': 112.65}
demand = {'C1': 216, 'C2': 216, 'C3': 216, 'C4': 144, 'C5': 144, 'C6': 144}
transport_cost = {('S1', 'C1'): 0.08, ('S1', 'C2'): 52.33, ('S1', 'C3'): 73.57, ('S1', 'C4'): 1237.33, ('S1', 'C5'): 0.07, ('S1', 'C6'): 112.16, ('S2', 'C1'): 46.02, ('S2', 'C2'): 175.23, ('S2', 'C3'): 2026.83, ('S2', 'C4'): 299.89, ('S2', 'C5'): 966.53, ('S2', 'C6'): 1590.42, ('S3', 'C1'): 1031.74, ('S3', 'C2'): 78.13, ('S3', 'C3'): 99.02, ('S3', 'C4'): 277.07, ('S3', 'C5'): 884.45, ('S3', 'C6'): 1800.86, ('S4', 'C1'): 868.75, ('S4', 'C2'): 94.2, ('S4', 'C3'): 1776.34, ('S4', 'C4'): 285.48, ('S4', 'C5'): 868.85, ('S4', 'C6'): 86.55, ('S5', 'C1'): 1577, ('S5', 'C2'): 760.15, ('S5', 'C3'): 2090.19, ('S5', 'C4'): 43.2, ('S5', 'C5'): 1577.12, ('S5', 'C6'): 1095.17, ('S6', 'C1'): 49.14, ('S6', 'C2'): 4.33, ('S6', 'C3'): 2079.57, ('S6', 'C4'): 277.04, ('S6', 'C5'): 1032.01, ('S6', 'C6'): 1543.49}
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}')
for i in suppliers:
    for j in stores:
        if (i, j) not in transport_cost:
            raise ValueError(f'Missing transportation cost for supplier {i} to store {j}')
m = Model()
m.Params.MIPGap = 0.0001
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
x_vars = m.addVars(suppliers, stores, vtype=GRB.CONTINUOUS, lb=0, name='')
for j in stores:
    m.addConstr(quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name='d_' + j)
for i in suppliers:
    for j in stores:
        m.addConstr(x_vars[i, j] <= demand[j] * y_vars[i], name='link_' + i + '_' + j)
m.setObjective(quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)) + quicksum((transport_cost[i, j] * x_vars[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')