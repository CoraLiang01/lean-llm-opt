import gurobipy as gp
from gurobipy import GRB

def solve_bandcamp_warehouse():
    warehouses = ['S1', 'S2', 'S3']
    customers = ['C1', 'C2', 'C3']
    fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
    demand = {'C1': 1083, 'C2': 776, 'C3': 16214}
    transportation_cost = {('S1', 'C1'): 1506.22, ('S1', 'C2'): 70.9, ('S1', 'C3'): 8.44, ('S2', 'C1'): 1732.65, ('S2', 'C2'): 1780.72, ('S2', 'C3'): 567.44, ('S3', 'C1'): 115.66, ('S3', 'C2'): 100.76, ('S3', 'C3'): 64.68}
    for w in warehouses:
        if w not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {w}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    for w in warehouses:
        for c in customers:
            if (w, c) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({w},{c})')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(warehouses, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(warehouses, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gp.LinExpr()
    for w in warehouses:
        obj += fixed_cost[w] * y[w]
    for w in warehouses:
        for c in customers:
            obj += transportation_cost[w, c] * x[w, c]
    m.setObjective(obj, GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((x[w, c] for w in warehouses)) == demand[c], name='demand_' + c)
    for w in warehouses:
        for c in customers:
            m.addConstr(x[w, c] <= demand[c] * y[w], name='link_' + w + '_' + c)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for w in warehouses:
            print(f'y[{w}] {y[w].X}')
        for w in warehouses:
            for c in customers:
                print(f'x[{w},{c}] {x[w, c].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_bandcamp_warehouse()