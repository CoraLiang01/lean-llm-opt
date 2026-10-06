import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = ['W%d' % i for i in range(1, 11)]
    J = ['C%d' % j for j in range(1, 21)]
    f_list = [2000, 2500, 1800, 3200, 1500, 4000, 2800, 1950, 3500, 2200]
    s_list = [1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 2100, 1300]
    d_list = [800, 600, 500, 700, 450, 950, 350, 850, 400, 750, 900, 550, 650, 820, 480, 920, 320, 780, 520, 680]
    c_matrix = [[10, 15, 20, 11, 16, 18, 7, 12, 22, 9, 14, 19, 25, 13, 17, 6, 21, 15, 8, 10], [18, 12, 9, 14, 10, 5, 19, 23, 11, 16, 20, 8, 15, 22, 7, 13, 24, 17, 12, 6], [13, 17, 15, 8, 12, 21, 16, 10, 5, 24, 13, 22, 7, 19, 14, 18, 9, 25, 11, 16], [7, 22, 11, 16, 20, 8, 15, 19, 13, 25, 6, 14, 21, 9, 23, 17, 10, 18, 24, 5], [16, 9, 25, 13, 7, 10, 23, 14, 18, 21, 5, 17, 9, 24, 12, 20, 6, 15, 19, 11], [22, 6, 14, 19, 23, 11, 8, 17, 9, 12, 15, 24, 5, 20, 10, 25, 13, 7, 18, 16], [8, 25, 17, 9, 14, 22, 11, 6, 16, 20, 18, 13, 24, 5, 19, 12, 23, 10, 7, 15], [19, 11, 7, 21, 15, 24, 13, 16, 20, 8, 17, 10, 12, 23, 5, 14, 22, 9, 16, 25], [12, 20, 5, 23, 17, 14, 9, 25, 18, 11, 16, 21, 10, 7, 24, 15, 19, 6, 13, 22], [25, 14, 22, 5, 19, 12, 24, 7, 15, 17, 23, 6, 16, 10, 20, 9, 18, 11, 25, 14]]
    f = {I[i]: f_list[i] for i in range(len(I))}
    s = {I[i]: s_list[i] for i in range(len(I))}
    d = {J[j]: d_list[j] for j in range(len(J))}
    c = {(I[i], J[j]): c_matrix[i][j] for i in range(len(I)) for j in range(len(J))}
    if len(f) != len(I):
        raise ValueError('Fixed cost vector f does not cover all warehouses.')
    if len(s) != len(I):
        raise ValueError('Capacity vector s does not cover all warehouses.')
    if len(d) != len(J):
        raise ValueError('Demand vector d does not cover all customers.')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Transportation cost c missing for ({i},{j})')
    m = gp.Model('facility_location')
    y = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == d[j], name='demand_%s' % j)
    for i in I:
        m.addConstr(gp.quicksum((x[i, j] for j in J)) <= s[i] * y[i], name='cap_%s' % i)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_problem()