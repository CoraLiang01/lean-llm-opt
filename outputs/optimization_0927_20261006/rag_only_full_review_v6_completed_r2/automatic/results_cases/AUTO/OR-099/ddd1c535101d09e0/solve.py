import gurobipy as gp
from gurobipy import GRB

def build_and_solve():
    I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    f = {1: 3000, 2: 3200, 3: 3100, 4: 2800, 5: 3500, 6: 2700, 7: 2900, 8: 3050, 9: 3100, 10: 2200, 11: 2890}
    cap = {1: 180, 2: 160, 3: 200, 4: 150, 5: 170, 6: 190, 7: 160, 8: 175, 9: 170, 10: 180, 11: 190}
    d = {1: 30, 2: 40, 3: 20, 4: 35, 5: 20, 6: 25, 7: 45, 8: 38, 9: 32, 10: 41, 11: 44}
    C_matrix = [[12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15], [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16], [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17], [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18], [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17], [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19], [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14], [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18], [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18], [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19], [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]]
    c = {}
    for (idx_i, i) in enumerate(I):
        for (idx_j, j) in enumerate(J):
            c[i, j] = C_matrix[idx_i][idx_j]
    if set(f.keys()) != set(I):
        raise ValueError('Mismatch in warehouse opening cost keys and I')
    if set(cap.keys()) != set(I):
        raise ValueError('Mismatch in warehouse capacity keys and I')
    if set(d.keys()) != set(J):
        raise ValueError('Mismatch in store demand keys and J')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Missing transportation cost for ({i},{j})')
    m = gp.Model('CFLP')
    y_vars = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y_vars[i] for i in I)) + gp.quicksum((c[i, j] * x_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) == d[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in J)) <= cap[i] * y_vars[i], name=f'cap_{i}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in I:
            print(f'y[{i}] {y_vars[i].X}')
        for i in I:
            for j in J:
                print(f'x[{i},{j}] {x_vars[i, j].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve()