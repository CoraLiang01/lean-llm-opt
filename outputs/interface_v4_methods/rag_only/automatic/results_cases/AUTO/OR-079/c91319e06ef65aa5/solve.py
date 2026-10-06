import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
    J = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    f = {'A1': 0, 'A2': 175, 'A3': 300, 'A4': 375, 'A5': 500, 'A6': 200, 'A7': 260, 'A8': 220, 'A9': 320, 'A10': 280, 'A11': 350, 'A12': 420, 'A13': 470, 'A14': 520, 'A15': 560}
    cap = {'A1': 30, 'A2': 10, 'A3': 20, 'A4': 30, 'A5': 40, 'A6': 20, 'A7': 25, 'A8': 30, 'A9': 35, 'A10': 20, 'A11': 40, 'A12': 25, 'A13': 30, 'A14': 50, 'A15': 45}
    d = {'B1': 30, 'B2': 25, 'B3': 20, 'B4': 35, 'B5': 25, 'B6': 30, 'B7': 25, 'B8': 30}
    c_matrix = [[8, 4, 3, 6, 7, 5, 9, 8], [5, 2, 3, 5, 6, 4, 7, 6], [4, 3, 4, 6, 5, 5, 6, 7], [9, 7, 5, 8, 9, 6, 10, 7], [10, 4, 2, 6, 8, 5, 7, 3], [6, 5, 4, 5, 7, 6, 8, 5], [7, 6, 5, 4, 6, 7, 9, 6], [5, 4, 6, 3, 5, 6, 7, 6], [8, 7, 6, 7, 9, 8, 10, 7], [6, 5, 7, 4, 6, 5, 7, 5], [9, 6, 4, 6, 8, 7, 9, 6], [7, 5, 6, 5, 6, 5, 8, 5], [8, 6, 5, 6, 7, 6, 8, 7], [9, 5, 3, 5, 7, 4, 6, 4], [10, 6, 4, 5, 8, 5, 7, 5]]
    c = {}
    for idx_i, i in enumerate(I):
        c[i] = {}
        for idx_j, j in enumerate(J):
            c[i][j] = c_matrix[idx_i][idx_j]
    if len(f) != len(I):
        raise ValueError('Fixed cost vector length does not match number of factories.')
    if len(cap) != len(I):
        raise ValueError('Capacity vector length does not match number of factories.')
    if len(d) != len(J):
        raise ValueError('Demand vector length does not match number of distribution centers.')
    if len(c_matrix) != len(I) or any((len(row) != len(J) for row in c_matrix)):
        raise ValueError('Shipping cost matrix dimensions do not match (factories x distribution centers).')
    for i in I:
        if i not in f or i not in cap or i not in c:
            raise ValueError(f'Missing data for factory {i}.')
        for j in J:
            if j not in c[i]:
                raise ValueError(f'Missing shipping cost for ({i},{j}).')
    for j in J:
        if j not in d:
            raise ValueError(f'Missing demand for distribution center {j}.')
    m = gp.Model('facility_location')
    y = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == d[j], name='demand_' + j)
    for i in I:
        m.addConstr(gp.quicksum((x[i, j] for j in J)) <= cap[i] * y[i], name='cap_' + i)
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