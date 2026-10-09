from gurobipy import Model, GRB, quicksum
I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
f = {1: 3000, 2: 3200, 3: 3100, 4: 2800, 5: 3500, 6: 2700, 7: 2900, 8: 3050, 9: 3100, 10: 2200, 11: 2890}
s = {1: 180, 2: 160, 3: 200, 4: 150, 5: 170, 6: 190, 7: 160, 8: 175, 9: 170, 10: 180, 11: 190}
d = {1: 30, 2: 40, 3: 20, 4: 35, 5: 20, 6: 25, 7: 45, 8: 38, 9: 32, 10: 41, 11: 44}
C_matrix = [[12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15], [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16], [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17], [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18], [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17], [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19], [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14], [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18], [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18], [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19], [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]]
c = {}
for (i_idx, i) in enumerate(I):
    for (j_idx, j) in enumerate(J):
        c[i, j] = C_matrix[i_idx][j_idx]
if set(f.keys()) != set(I):
    raise ValueError('Mismatch in warehouse opening cost keys and I')
if set(s.keys()) != set(I):
    raise ValueError('Mismatch in warehouse capacity keys and I')
if set(d.keys()) != set(J):
    raise ValueError('Mismatch in store demand keys and J')
for i in I:
    for j in J:
        if (i, j) not in c:
            raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')

def build_model():
    model = Model()
    model.Params.MIPGap = 0.0001
    y_vars = model.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = model.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    model.setObjective(quicksum((f[i] * y_vars[i] for i in I)) + quicksum((c[i, j] * x_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        model.addConstr(quicksum((x_vars[i, j] for i in I)) == d[j], name='')
    for i in I:
        model.addConstr(quicksum((x_vars[i, j] for j in J)) <= s[i] * y_vars[i], name='')
    return model
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)