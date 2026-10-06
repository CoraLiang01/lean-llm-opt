import gurobipy as gp
from gurobipy import GRB

def solve_superstore_optimization():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5']
    branches = ['C1', 'C2', 'C3', 'C4', 'C5']
    f = {'S1': 97.65, 'S2': 99.76, 'S3': 100.76, 'S4': 105.32, 'S5': 98.88}
    d = {'C1': 143, 'C2': 6, 'C3': 10, 'C4': 25, 'C5': 3}
    c = {'S1': {'C1': 150.74, 'C2': 0.02, 'C3': 49.13, 'C4': 2080.15, 'C5': 426.4}, 'S2': {'C1': 233.05, 'C2': 97.73, 'C3': 49.84, 'C4': 1982.39, 'C5': 23.96}, 'S3': {'C1': 55.68, 'C2': 935.61, 'C3': 4.03, 'C4': 73.09, 'C5': 525.32}, 'S4': {'C1': 1483.82, 'C2': 1801.08, 'C3': 112.16, 'C4': 816.05, 'C5': 107.01}, 'S5': {'C1': 1119.47, 'C2': 884.31, 'C3': 0.08, 'C4': 1544.95, 'C5': 543.67}}
    if set(f.keys()) != set(suppliers):
        raise ValueError('Fixed cost keys do not match supplier set.')
    if set(d.keys()) != set(branches):
        raise ValueError('Demand keys do not match branch set.')
    if set(c.keys()) != set(suppliers):
        raise ValueError('Transportation cost keys do not match supplier set.')
    for i in suppliers:
        if set(c[i].keys()) != set(branches):
            raise ValueError(f'Transportation cost for supplier {i} does not cover all branches.')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(suppliers, branches, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in suppliers)) + gp.quicksum((c[i][j] * x[i, j] for i in suppliers for j in branches)), GRB.MINIMIZE)
    for j in branches:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == d[j], name='demand_' + j)
    for i in suppliers:
        for j in branches:
            m.addConstr(x[i, j] <= d[j] * y[i], name='link_' + i + '_' + j)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_superstore_optimization()