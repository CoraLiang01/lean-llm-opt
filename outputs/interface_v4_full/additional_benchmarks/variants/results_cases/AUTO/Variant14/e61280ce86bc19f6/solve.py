import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    J = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10']
    f = {'B1': 11, 'B2': 14, 'B3': 10, 'B4': 13, 'B5': 16, 'B6': 9, 'B7': 12, 'B8': 15}
    C = {'B1': ['Z1', 'Z2', 'Z5'], 'B2': ['Z2', 'Z3', 'Z6'], 'B3': ['Z4', 'Z5', 'Z8'], 'B4': ['Z1', 'Z6', 'Z7'], 'B5': ['Z3', 'Z7', 'Z9'], 'B6': ['Z8', 'Z9', 'Z10'], 'B7': ['Z4', 'Z10'], 'B8': ['Z5', 'Z6', 'Z9']}
    I_j = {'Z1': ['B1', 'B4'], 'Z2': ['B1', 'B2'], 'Z3': ['B2', 'B5'], 'Z4': ['B3', 'B7'], 'Z5': ['B1', 'B3', 'B8'], 'Z6': ['B2', 'B4', 'B8'], 'Z7': ['B4', 'B5'], 'Z8': ['B3', 'B6'], 'Z9': ['B5', 'B6', 'B8'], 'Z10': ['B6', 'B7']}
    if set(f.keys()) != set(I):
        raise ValueError('Opening cost data missing for some depots.')
    for i in I:
        if i not in C:
            raise ValueError(f'Coverage set missing for depot {i}.')
    for j in J:
        if j not in I_j:
            raise ValueError(f'Depot coverage list missing for zone {j}.')
        if not set(I_j[j]).issubset(set(I)):
            raise ValueError(f'Depot(s) in I_j[{j}] not in I.')
    m = gp.Model('SetCoveringDepots')
    m.setParam('MIPGap', 0.0001)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((y[i] for i in I_j[j])) >= 1 for j in J), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()