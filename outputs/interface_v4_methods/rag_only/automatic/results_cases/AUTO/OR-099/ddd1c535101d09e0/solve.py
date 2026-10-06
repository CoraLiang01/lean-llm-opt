import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = list(range(1, 12))
    J = list(range(1, 12))
    f_i = {1: 3000, 2: 3200, 3: 3100, 4: 2800, 5: 3500, 6: 2700, 7: 2900, 8: 3050, 9: 3100, 10: 2200, 11: 2890}
    s_i = {1: 180, 2: 160, 3: 200, 4: 150, 5: 170, 6: 190, 7: 160, 8: 175, 9: 170, 10: 180, 11: 190}
    d_j = {1: 30, 2: 40, 3: 20, 4: 35, 5: 20, 6: 25, 7: 45, 8: 38, 9: 32, 10: 41, 11: 44}
    c_ij_matrix = [[12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15], [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16], [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17], [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18], [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17], [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19], [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14], [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18], [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18], [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19], [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]]
    c_ij = {}
    for idx_i, i in enumerate(I):
        c_ij[i] = {}
        for idx_j, j in enumerate(J):
            c_ij[i][j] = c_ij_matrix[idx_i][idx_j]
    if set(f_i.keys()) != set(I):
        raise ValueError('f_i keys do not match I')
    if set(s_i.keys()) != set(I):
        raise ValueError('s_i keys do not match I')
    if set(d_j.keys()) != set(J):
        raise ValueError('d_j keys do not match J')
    if set(c_ij.keys()) != set(I):
        raise ValueError('c_ij keys do not match I')
    for i in I:
        if set(c_ij[i].keys()) != set(J):
            raise ValueError(f'c_ij[{i}] keys do not match J')
    m = gp.Model('warehouse_location')
    m.Params.MIPGap = 0.0001
    y = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f_i[i] * y[i] for i in I)) + gp.quicksum((c_ij[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == d_j[j], name='demand')
    for i in I:
        m.addConstr(gp.quicksum((x[i, j] for j in J)) <= s_i[i] * y[i], name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in I:
            print(f'y[{i}] {y[i].X}')
        for i in I:
            for j in J:
                print(f'x[{i},{j}] {x[i, j].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()