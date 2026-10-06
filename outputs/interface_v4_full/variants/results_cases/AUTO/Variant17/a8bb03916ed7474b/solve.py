import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']
    J = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12']
    d = {'A1': 25, 'A2': 35, 'A3': 40, 'A4': 30, 'A5': 50, 'A6': 45, 'A7': 20, 'A8': 55, 'A9': 60, 'A10': 30, 'A11': 42, 'A12': 38}
    C_matrix = [[2, 3, 4, 8, 9, 10, 13, 14, 15, 12, 11, 10], [3, 2, 3, 7, 8, 9, 12, 13, 14, 11, 10, 9], [8, 7, 5, 2, 3, 4, 8, 9, 11, 7, 6, 7], [9, 8, 6, 3, 2, 3, 7, 8, 10, 6, 5, 6], [13, 12, 10, 8, 7, 6, 2, 3, 4, 5, 6, 7], [14, 13, 11, 9, 8, 7, 3, 2, 3, 4, 5, 6], [11, 10, 8, 7, 6, 5, 6, 5, 4, 2, 3, 2]]
    c = {}
    for idx_i, i in enumerate(I):
        c[i] = {}
        for idx_j, j in enumerate(J):
            c[i][j] = C_matrix[idx_i][idx_j]
    p = 3
    if set(d.keys()) != set(J):
        raise ValueError('Demand data missing or extra areas.')
    for i in I:
        if i not in c or set(c[i].keys()) != set(J):
            raise ValueError(f'Distance data missing or extra for location {i}.')
    m = gp.Model('p_median')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, J, vtype=GRB.BINARY, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((d[j] * c[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == 1 for j in J), name='')
    m.addConstr(gp.quicksum((y[i] for i in I)) == p, name='open_facilities')
    m.addConstrs((x[i, j] <= y[i] for i in I for j in J), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()