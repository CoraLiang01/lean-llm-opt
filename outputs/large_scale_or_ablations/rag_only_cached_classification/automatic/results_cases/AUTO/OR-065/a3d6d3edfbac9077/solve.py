import gurobipy as gp
from gurobipy import GRB

def solve_bandcamp_warehouse():
    F = ['S1', 'S2', 'S3']
    C = ['C1', 'C2', 'C3']
    f = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
    d = {'C1': 1083, 'C2': 776, 'C3': 16214}
    c = {'S1': {'C1': 1506.22, 'C2': 70.9, 'C3': 8.44}, 'S2': {'C1': 1732.65, 'C2': 1780.72, 'C3': 567.44}, 'S3': {'C1': 115.66, 'C2': 100.76, 'C3': 64.68}}
    if set(f.keys()) != set(F):
        raise ValueError('Fixed cost vector f keys do not match warehouse set F.')
    if set(d.keys()) != set(C):
        raise ValueError('Demand vector d keys do not match customer set C.')
    if set(c.keys()) != set(F):
        raise ValueError('Transportation cost matrix c keys do not match warehouse set F.')
    for i in F:
        if set(c[i].keys()) != set(C):
            raise ValueError(f'Transportation cost matrix c[{i}] keys do not match customer set C.')
    m = gp.Model('bandcamp_warehouse')
    y = m.addVars(F, vtype=GRB.BINARY, name='')
    x = m.addVars(F, C, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in F)) + gp.quicksum((c[i][j] * x[i, j] for i in F for j in C)), GRB.MINIMIZE)
    for j in C:
        m.addConstr(gp.quicksum((x[i, j] for i in F)) == d[j], name='')
    for i in F:
        for j in C:
            m.addConstr(x[i, j] <= d[j] * y[i], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal:.6f}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X:.6f}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_bandcamp_warehouse()