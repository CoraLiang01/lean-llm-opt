import gurobipy as gp
from gurobipy import GRB

def solve_electrotech_network():
    factories = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
    dcs = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    f = {'A1': 0, 'A2': 175, 'A3': 300, 'A4': 375, 'A5': 500, 'A6': 200, 'A7': 260, 'A8': 220, 'A9': 320, 'A10': 280, 'A11': 350, 'A12': 420, 'A13': 470, 'A14': 520, 'A15': 560}
    cap = {'A1': 30, 'A2': 10, 'A3': 20, 'A4': 30, 'A5': 40, 'A6': 20, 'A7': 25, 'A8': 30, 'A9': 35, 'A10': 20, 'A11': 40, 'A12': 25, 'A13': 30, 'A14': 50, 'A15': 45}
    d = {'B1': 30, 'B2': 25, 'B3': 20, 'B4': 35, 'B5': 25, 'B6': 30, 'B7': 25, 'B8': 30}
    C_matrix = [[8, 4, 3, 6, 7, 5, 9, 8], [5, 2, 3, 5, 6, 4, 7, 6], [4, 3, 4, 6, 5, 5, 6, 7], [9, 7, 5, 8, 9, 6, 10, 7], [10, 4, 2, 6, 8, 5, 7, 3], [6, 5, 4, 5, 7, 6, 8, 5], [7, 6, 5, 4, 6, 7, 9, 6], [5, 4, 6, 3, 5, 6, 7, 6], [8, 7, 6, 7, 9, 8, 10, 7], [6, 5, 7, 4, 6, 5, 7, 5], [9, 6, 4, 6, 8, 7, 9, 6], [7, 5, 6, 5, 6, 5, 8, 5], [8, 6, 5, 6, 7, 6, 8, 7], [9, 5, 3, 5, 7, 4, 6, 4], [10, 6, 4, 5, 8, 5, 7, 5]]
    C = {}
    for (i_idx, i) in enumerate(factories):
        C[i] = {}
        for (j_idx, j) in enumerate(dcs):
            C[i][j] = C_matrix[i_idx][j_idx]
    if set(f.keys()) != set(factories):
        raise ValueError('Mismatch in factory fixed cost keys and factories set')
    if set(cap.keys()) != set(factories):
        raise ValueError('Mismatch in factory capacity keys and factories set')
    if set(d.keys()) != set(dcs):
        raise ValueError('Mismatch in demand keys and distribution centers set')
    if len(C) != len(factories) or any((len(C[i]) != len(dcs) for i in factories)):
        raise ValueError('Mismatch in shipping cost matrix dimensions')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(factories, vtype=GRB.BINARY, name='')
    x = m.addVars(factories, dcs, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in factories)) + gp.quicksum((C[i][j] * x[i, j] for i in factories for j in dcs)), GRB.MINIMIZE)
    for j in dcs:
        m.addConstr(gp.quicksum((x[i, j] for i in factories)) == d[j], name='')
    for i in factories:
        m.addConstr(gp.quicksum((x[i, j] for j in dcs)) <= cap[i] * y[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_electrotech_network()