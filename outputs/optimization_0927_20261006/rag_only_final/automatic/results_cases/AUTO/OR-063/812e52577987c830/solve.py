from gurobipy import Model, GRB, quicksum
F = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
C = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83, 'S4': 98.71, 'S5': 95.73, 'S6': 99.96, 'S7': 98.16}
demand = {'C1': 1083, 'C2': 776, 'C3': 16214, 'C4': 553, 'C5': 17106, 'C6': 594, 'C7': 732}
transportation_cost = {('S1', 'C1'): 1506.22, ('S1', 'C2'): 70.9, ('S1', 'C3'): 8.44, ('S1', 'C4'): 260.27, ('S1', 'C5'): 197.47, ('S1', 'C6'): 71.71, ('S1', 'C7'): 61.19, ('S2', 'C1'): 1732.65, ('S2', 'C2'): 1780.72, ('S2', 'C3'): 567.44, ('S2', 'C4'): 448.68, ('S2', 'C5'): 29.0, ('S2', 'C6'): 1484.91, ('S2', 'C7'): 963.92, ('S3', 'C1'): 115.66, ('S3', 'C2'): 100.76, ('S3', 'C3'): 64.68, ('S3', 'C4'): 1324.53, ('S3', 'C5'): 64.99, ('S3', 'C6'): 134.88, ('S3', 'C7'): 2102.83, ('S4', 'C1'): 1254.78, ('S4', 'C2'): 1115.63, ('S4', 'C3'): 52.31, ('S4', 'C4'): 1036.16, ('S4', 'C5'): 892.63, ('S4', 'C6'): 1464.04, ('S4', 'C7'): 1383.41, ('S5', 'C1'): 42.9, ('S5', 'C2'): 891.01, ('S5', 'C3'): 1013.94, ('S5', 'C4'): 1128.72, ('S5', 'C5'): 58.91, ('S5', 'C6'): 42.89, ('S5', 'C7'): 1570.31, ('S6', 'C1'): 0.7, ('S6', 'C2'): 139.46, ('S6', 'C3'): 70.03, ('S6', 'C4'): 79.15, ('S6', 'C5'): 1482.0, ('S6', 'C6'): 0.91, ('S6', 'C7'): 110.46, ('S7', 'C1'): 1732.3, ('S7', 'C2'): 1780.44, ('S7', 'C3'): 486.5, ('S7', 'C4'): 523.74, ('S7', 'C5'): 522.08, ('S7', 'C6'): 82.48, ('S7', 'C7'): 826.41}
for i in F:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for warehouse {i}')
for j in C:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
for i in F:
    for j in C:
        if (i, j) not in transportation_cost:
            raise ValueError(f'Missing transportation cost for ({i},{j})')

def build_model():
    m = Model()
    m.setParam('MIPGap', 0.0001)
    y_vars = m.addVars(F, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(F, C, vtype=GRB.CONTINUOUS, lb=0, name='')
    for j in C:
        m.addConstr(quicksum((x_vars[i, j] for i in F)) == demand[j], name='demand_' + j)
    for i in F:
        for j in C:
            m.addConstr(x_vars[i, j] <= demand[j] * y_vars[i], name='link_%s_%s' % (i, j))
    m.setObjective(quicksum((fixed_cost[i] * y_vars[i] for i in F)) + quicksum((transportation_cost[i, j] * x_vars[i, j] for i in F for j in C)), GRB.MINIMIZE)
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')