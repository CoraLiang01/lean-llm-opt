import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    F = ['S1', 'S2', 'S3']
    C = ['C1', 'C2', 'C3']
    fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
    transportation_cost = {('S1', 'C1'): 1506.22, ('S1', 'C2'): 70.9, ('S1', 'C3'): 8.44, ('S2', 'C1'): 1732.65, ('S2', 'C2'): 1780.72, ('S2', 'C3'): 567.44, ('S3', 'C1'): 115.66, ('S3', 'C2'): 100.76, ('S3', 'C3'): 64.68}
    demand = {'C1': 1083, 'C2': 776, 'C3': 16214}
    if set(fixed_cost.keys()) != set(F):
        raise ValueError('fixed_cost keys do not match F')
    if set(demand.keys()) != set(C):
        raise ValueError('demand keys do not match C')
    if set(transportation_cost.keys()) != set(((i, j) for i in F for j in C)):
        raise ValueError('transportation_cost keys do not match F x C')
    m = gp.Model('bandcamp_warehouse')
    y = m.addVars(F, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(F, C, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in F)) + gp.quicksum((transportation_cost[i, j] * x[i, j] for i in F for j in C)), GRB.MINIMIZE)
    for j in C:
        m.addConstr(gp.quicksum((x[i, j] for i in F)) == demand[j], name='demand_' + j)
    for i in F:
        for j in C:
            m.addConstr(x[i, j] <= demand[j] * y[i], name='link_%s_%s' % (i, j))
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