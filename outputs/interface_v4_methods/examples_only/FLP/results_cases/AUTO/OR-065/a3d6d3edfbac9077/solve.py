import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    F = ['S1', 'S2', 'S3']
    C = ['C1', 'C2', 'C3']
    fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
    transportation_cost = {('S1', 'C1'): 1506.22, ('S1', 'C2'): 70.9, ('S1', 'C3'): 8.44, ('S2', 'C1'): 1732.65, ('S2', 'C2'): 1780.72, ('S2', 'C3'): 567.44, ('S3', 'C1'): 115.66, ('S3', 'C2'): 100.76, ('S3', 'C3'): 64.68}
    demand = {'C1': 1083, 'C2': 776, 'C3': 16214}
    for f in F:
        if f not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {f}')
    for c in C:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    for f in F:
        for c in C:
            if (f, c) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({f},{c})')
    m = gp.Model('facility_location')
    y = m.addVars(F, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(F, C, vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gp.LinExpr()
    for f in F:
        obj += fixed_cost[f] * y[f]
    for f in F:
        for c in C:
            obj += transportation_cost[f, c] * x[f, c]
    m.setObjective(obj, GRB.MINIMIZE)
    for c in C:
        m.addConstr(gp.quicksum((x[f, c] for f in F)) == demand[c], name=f'demand_{c}')
    for f in F:
        for c in C:
            m.addConstr(x[f, c] <= demand[c] * y[f], name=f'link_{f}_{c}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()