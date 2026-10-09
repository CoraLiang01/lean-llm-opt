from gurobipy import Model, GRB

def solve_bandcamp_warehouse():
    F = ['S1', 'S2', 'S3']
    C = ['C1', 'C2', 'C3']
    fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
    demand = {'C1': 1083, 'C2': 776, 'C3': 16214}
    transportation_cost = {('S1', 'C1'): 1506.22, ('S1', 'C2'): 70.9, ('S1', 'C3'): 8.44, ('S2', 'C1'): 1732.65, ('S2', 'C2'): 1780.72, ('S2', 'C3'): 567.44, ('S3', 'C1'): 115.66, ('S3', 'C2'): 100.76, ('S3', 'C3'): 64.68}
    for i in F:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {i}')
    for j in C:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in F:
        for j in C:
            if (i, j) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({i},{j})')
    m = Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(F, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(F, C, vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = sum((fixed_cost[i] * y[i] for i in F)) + sum((transportation_cost[i, j] * x[i, j] for i in F for j in C))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in C:
        m.addConstr(sum((x[i, j] for i in F)) == demand[j], name=f'demand_{j}')
    for i in F:
        for j in C:
            m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_bandcamp_warehouse()