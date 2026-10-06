import gurobipy as gp
from gurobipy import GRB

def solve_superstore_supplier_selection():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5']
    branches = ['C1', 'C2', 'C3', 'C4', 'C5']
    fixed_cost = {'S1': 97.65, 'S2': 99.76, 'S3': 100.76, 'S4': 105.32, 'S5': 98.88}
    demand = {'C1': 143, 'C2': 6, 'C3': 10, 'C4': 25, 'C5': 3}
    transportation_cost = {'S1': {'C1': 150.74, 'C2': 0.02, 'C3': 49.13, 'C4': 2080.15, 'C5': 426.4}, 'S2': {'C1': 233.05, 'C2': 97.73, 'C3': 49.84, 'C4': 1982.39, 'C5': 23.96}, 'S3': {'C1': 55.68, 'C2': 935.61, 'C3': 4.03, 'C4': 73.09, 'C5': 525.32}, 'S4': {'C1': 1483.82, 'C2': 1801.08, 'C3': 112.16, 'C4': 816.05, 'C5': 107.01}, 'S5': {'C1': 1119.47, 'C2': 884.31, 'C3': 0.08, 'C4': 1544.95, 'C5': 543.67}}
    if set(fixed_cost.keys()) != set(suppliers):
        raise ValueError('Fixed cost data missing or extra suppliers.')
    if set(demand.keys()) != set(branches):
        raise ValueError('Demand data missing or extra branches.')
    if set(transportation_cost.keys()) != set(suppliers):
        raise ValueError('Transportation cost data missing or extra suppliers.')
    for i in suppliers:
        if set(transportation_cost[i].keys()) != set(branches):
            raise ValueError(f'Transportation cost data for supplier {i} missing or extra branches.')
    m = gp.Model('superstore_supplier_selection')
    y = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(suppliers, branches, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_cost[i][j] * x[i, j] for i in suppliers for j in branches)), GRB.MINIMIZE)
    for j in branches:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
    for i in suppliers:
        for j in branches:
            m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal:.6f}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X:.6f}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_superstore_supplier_selection()