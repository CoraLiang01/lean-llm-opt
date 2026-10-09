from gurobipy import Model, GRB, quicksum
factory_ids = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
dc_ids = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
f = {'A1': 0, 'A2': 175, 'A3': 300, 'A4': 375, 'A5': 500, 'A6': 200, 'A7': 260, 'A8': 220, 'A9': 320, 'A10': 280, 'A11': 350, 'A12': 420, 'A13': 470, 'A14': 520, 'A15': 560}
cap = {'A1': 30, 'A2': 10, 'A3': 20, 'A4': 30, 'A5': 40, 'A6': 20, 'A7': 25, 'A8': 30, 'A9': 35, 'A10': 20, 'A11': 40, 'A12': 25, 'A13': 30, 'A14': 50, 'A15': 45}
d = {'B1': 30, 'B2': 25, 'B3': 20, 'B4': 35, 'B5': 25, 'B6': 30, 'B7': 25, 'B8': 30}
C_matrix = [[8, 4, 3, 6, 7, 5, 9, 8], [5, 2, 3, 5, 6, 4, 7, 6], [4, 3, 4, 6, 5, 5, 6, 7], [9, 7, 5, 8, 9, 6, 10, 7], [10, 4, 2, 6, 8, 5, 7, 3], [6, 5, 4, 5, 7, 6, 8, 5], [7, 6, 5, 4, 6, 7, 9, 6], [5, 4, 6, 3, 5, 6, 7, 6], [8, 7, 6, 7, 9, 8, 10, 7], [6, 5, 7, 4, 6, 5, 7, 5], [9, 6, 4, 6, 8, 7, 9, 6], [7, 5, 6, 5, 6, 5, 8, 5], [8, 6, 5, 6, 7, 6, 8, 7], [9, 5, 3, 5, 7, 4, 6, 4], [10, 6, 4, 5, 8, 5, 7, 5]]
c = {}
for (i_idx, i) in enumerate(factory_ids):
    for (j_idx, j) in enumerate(dc_ids):
        c[i, j] = C_matrix[i_idx][j_idx]
if set(f.keys()) != set(factory_ids):
    raise ValueError('Fixed cost data missing or extra factories.')
if set(cap.keys()) != set(factory_ids):
    raise ValueError('Capacity data missing or extra factories.')
if set(d.keys()) != set(dc_ids):
    raise ValueError('Demand data missing or extra DCs.')
for i in factory_ids:
    for j in dc_ids:
        if (i, j) not in c:
            raise ValueError(f'Shipping cost missing for ({i},{j})')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(factory_ids, vtype=GRB.BINARY, name='')
    x_vars = m.addVars(factory_ids, dc_ids, vtype=GRB.CONTINUOUS, lb=0, name='')
    for j in dc_ids:
        m.addConstr(quicksum((x_vars[i, j] for i in factory_ids)) == d[j], name='')
    for i in factory_ids:
        m.addConstr(quicksum((x_vars[i, j] for j in dc_ids)) <= cap[i] * y_vars[i], name='')
    m.setObjective(quicksum((f[i] * y_vars[i] for i in factory_ids)) + quicksum((c[i, j] * x_vars[i, j] for i in factory_ids for j in dc_ids)), GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_model()