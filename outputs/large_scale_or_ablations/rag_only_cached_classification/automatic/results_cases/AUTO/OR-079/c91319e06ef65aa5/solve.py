import gurobipy as gp
from gurobipy import GRB

def solve_cflp():
    I = ['A%d' % i for i in range(1, 16)]
    J = ['B%d' % j for j in range(1, 9)]
    f_list = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560]
    cap_list = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45]
    d_list = [30, 25, 20, 35, 25, 30, 25, 30]
    c_matrix = [[8, 4, 3, 6, 7, 5, 9, 8], [5, 2, 3, 5, 6, 4, 7, 6], [4, 3, 4, 6, 5, 5, 6, 7], [9, 7, 5, 8, 9, 6, 10, 7], [10, 4, 2, 6, 8, 5, 7, 3], [6, 5, 4, 5, 7, 6, 8, 5], [7, 6, 5, 4, 6, 7, 9, 6], [5, 4, 6, 3, 5, 6, 7, 6], [8, 7, 6, 7, 9, 8, 10, 7], [6, 5, 7, 4, 6, 5, 7, 5], [9, 6, 4, 6, 8, 7, 9, 6], [7, 5, 6, 5, 6, 5, 8, 5], [8, 6, 5, 6, 7, 6, 8, 7], [9, 5, 3, 5, 7, 4, 6, 4], [10, 6, 4, 5, 8, 5, 7, 5]]
    f = {I[i]: f_list[i] for i in range(len(I))}
    cap = {I[i]: cap_list[i] for i in range(len(I))}
    d = {J[j]: d_list[j] for j in range(len(J))}
    c = {(I[i], J[j]): c_matrix[i][j] for i in range(len(I)) for j in range(len(J))}
    if len(f) != len(I):
        raise ValueError('Fixed cost vector length does not match factory set.')
    if len(cap) != len(I):
        raise ValueError('Capacity vector length does not match factory set.')
    if len(d) != len(J):
        raise ValueError('Demand vector length does not match distribution center set.')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Missing shipping cost for ({i},{j})')
    m = gp.Model('CFLP')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == d[j], name='')
    for i in I:
        m.addConstr(gp.quicksum((x[i, j] for j in J)) <= cap[i] * y[i], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_cflp()