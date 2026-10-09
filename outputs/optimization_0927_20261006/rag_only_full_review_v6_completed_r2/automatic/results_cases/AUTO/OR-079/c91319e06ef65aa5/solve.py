from gurobipy import Model, GRB, quicksum
I = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
J = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
f = {'A1': 0, 'A2': 175, 'A3': 300, 'A4': 375, 'A5': 500, 'A6': 200, 'A7': 260, 'A8': 220, 'A9': 320, 'A10': 280, 'A11': 350, 'A12': 420, 'A13': 470, 'A14': 520, 'A15': 560}
cap = {'A1': 30, 'A2': 10, 'A3': 20, 'A4': 30, 'A5': 40, 'A6': 20, 'A7': 25, 'A8': 30, 'A9': 35, 'A10': 20, 'A11': 40, 'A12': 25, 'A13': 30, 'A14': 50, 'A15': 45}
d = {'B1': 30, 'B2': 25, 'B3': 20, 'B4': 35, 'B5': 25, 'B6': 30, 'B7': 25, 'B8': 30}
C_rows = [[8, 4, 3, 6, 7, 5, 9, 8], [5, 2, 3, 5, 6, 4, 7, 6], [4, 3, 4, 6, 5, 5, 6, 7], [9, 7, 5, 8, 9, 6, 10, 7], [10, 4, 2, 6, 8, 5, 7, 3], [6, 5, 4, 5, 7, 6, 8, 5], [7, 6, 5, 4, 6, 7, 9, 6], [5, 4, 6, 3, 5, 6, 7, 6], [8, 7, 6, 7, 9, 8, 10, 7], [6, 5, 7, 4, 6, 5, 7, 5], [9, 6, 4, 6, 8, 7, 9, 6], [7, 5, 6, 5, 6, 5, 8, 5], [8, 6, 5, 6, 7, 6, 8, 7], [9, 5, 3, 5, 7, 4, 6, 4], [10, 6, 4, 5, 8, 5, 7, 5]]
c = {}
for (i_idx, i) in enumerate(I):
    for (j_idx, j) in enumerate(J):
        c[i, j] = C_rows[i_idx][j_idx]
if set(f.keys()) != set(I):
    raise ValueError('Facility cost keys do not match factory set I')
if set(cap.keys()) != set(I):
    raise ValueError('Capacity keys do not match factory set I')
if set(d.keys()) != set(J):
    raise ValueError('Demand keys do not match distribution center set J')
for i in I:
    for j in J:
        if (i, j) not in c:
            raise ValueError(f'Missing shipping cost for ({i},{j})')

def build_model():
    m = Model()
    m.setParam('MIPGap', 0.0001)
    y_vars = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((f[i] * y_vars[i] for i in I)) + quicksum((c[i, j] * x_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(quicksum((x_vars[i, j] for i in I)) == d[j], name='demand_' + j)
    for i in I:
        m.addConstr(quicksum((x_vars[i, j] for j in J)) <= cap[i] * y_vars[i], name='cap_' + i)
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)