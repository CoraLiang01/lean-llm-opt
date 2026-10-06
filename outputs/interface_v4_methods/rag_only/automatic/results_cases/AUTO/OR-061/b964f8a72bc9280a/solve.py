import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5']
    branches = ['C1', 'C2', 'C3', 'C4', 'C5']
    fixed_cost = {'S1': 97.65, 'S2': 99.76, 'S3': 100.76, 'S4': 105.32, 'S5': 98.88}
    demand = {'C1': 143, 'C2': 6, 'C3': 10, 'C4': 25, 'C5': 3}
    transportation_cost = {('S1', 'C1'): 150.74, ('S1', 'C2'): 0.02, ('S1', 'C3'): 49.13, ('S1', 'C4'): 2080.15, ('S1', 'C5'): 426.4, ('S2', 'C1'): 233.05, ('S2', 'C2'): 97.73, ('S2', 'C3'): 49.84, ('S2', 'C4'): 1982.39, ('S2', 'C5'): 23.96, ('S3', 'C1'): 55.68, ('S3', 'C2'): 935.61, ('S3', 'C3'): 4.03, ('S3', 'C4'): 73.09, ('S3', 'C5'): 525.32, ('S4', 'C1'): 1483.82, ('S4', 'C2'): 1801.08, ('S4', 'C3'): 112.16, ('S4', 'C4'): 816.05, ('S4', 'C5'): 107.01, ('S5', 'C1'): 1119.47, ('S5', 'C2'): 884.31, ('S5', 'C3'): 0.08, ('S5', 'C4'): 1544.95, ('S5', 'C5'): 543.67}
    if set(fixed_cost.keys()) != set(suppliers):
        raise ValueError('Fixed cost data missing or extra suppliers.')
    if set(demand.keys()) != set(branches):
        raise ValueError('Demand data missing or extra branches.')
    for i in suppliers:
        for j in branches:
            if (i, j) not in transportation_cost:
                raise ValueError(f'Transportation cost missing for ({i},{j})')
    total_demand = sum((demand[j] for j in branches))
    M = {i: total_demand for i in suppliers}
    m = gp.Model('superstore_inventory')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars(suppliers, branches, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_cost[i, j] * x[i, j] for i in suppliers for j in branches)), GRB.MINIMIZE)
    for j in branches:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name='')
    for i in suppliers:
        m.addConstr(gp.quicksum((x[i, j] for j in branches)) <= M[i] * y[i], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()