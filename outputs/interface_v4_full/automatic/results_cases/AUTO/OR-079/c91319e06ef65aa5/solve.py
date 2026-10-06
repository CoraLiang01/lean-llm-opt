import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
    J = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    f_list = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560]
    K_list = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45]
    d_list = [30, 25, 20, 35, 25, 30, 25, 30]
    C_matrix = [[8, 4, 3, 6, 7, 5, 9, 8], [5, 2, 3, 5, 6, 4, 7, 6], [4, 3, 4, 6, 5, 5, 6, 7], [9, 7, 5, 8, 9, 6, 10, 7], [10, 4, 2, 6, 8, 5, 7, 3], [6, 5, 4, 5, 7, 6, 8, 5], [7, 6, 5, 4, 6, 7, 9, 6], [5, 4, 6, 3, 5, 6, 7, 6], [8, 7, 6, 7, 9, 8, 10, 7], [6, 5, 7, 4, 6, 5, 7, 5], [9, 6, 4, 6, 8, 7, 9, 6], [7, 5, 6, 5, 6, 5, 8, 5], [8, 6, 5, 6, 7, 6, 8, 7], [9, 5, 3, 5, 7, 4, 6, 4], [10, 6, 4, 5, 8, 5, 7, 5]]
    f = {I[i]: f_list[i] for i in range(len(I))}
    K = {I[i]: K_list[i] for i in range(len(I))}
    d = {J[j]: d_list[j] for j in range(len(J))}
    c = {I[i]: {J[j]: C_matrix[i][j] for j in range(len(J))} for i in range(len(I))}
    if not (set(f.keys()) == set(I) and set(K.keys()) == set(I)):
        raise ValueError('Factory fixed cost or capacity data missing for some factories.')
    if not set(d.keys()) == set(J):
        raise ValueError('Demand data missing for some distribution centers.')
    for i in I:
        if set(c[i].keys()) != set(J):
            raise ValueError(f'Shipping cost data missing for some distribution centers for factory {i}.')
    m = gp.Model('ElectroTech_Facility_Location')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == d[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= K[i] * y[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()