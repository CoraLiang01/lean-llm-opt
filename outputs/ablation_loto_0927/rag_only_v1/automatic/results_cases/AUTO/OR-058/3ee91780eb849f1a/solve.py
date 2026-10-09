import gurobipy as gp
from gurobipy import GRB

def solve_supplier_store_optimization():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    stores = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    fixed_cost = {'S1': 98.88, 'S2': 99.73, 'S3': 94.01, 'S4': 93.77, 'S5': 107.59, 'S6': 112.65}
    demand = {'C1': 216, 'C2': 216, 'C3': 216, 'C4': 144, 'C5': 144, 'C6': 144}
    transportation_cost = {('S1', 'C1'): 0.08, ('S1', 'C2'): 52.33, ('S1', 'C3'): 73.57, ('S1', 'C4'): 1237.33, ('S1', 'C5'): 0.07, ('S1', 'C6'): 112.16, ('S2', 'C1'): 46.02, ('S2', 'C2'): 175.23, ('S2', 'C3'): 2026.83, ('S2', 'C4'): 299.89, ('S2', 'C5'): 966.53, ('S2', 'C6'): 1590.42, ('S3', 'C1'): 1031.74, ('S3', 'C2'): 78.13, ('S3', 'C3'): 99.02, ('S3', 'C4'): 277.07, ('S3', 'C5'): 884.45, ('S3', 'C6'): 1800.86, ('S4', 'C1'): 868.75, ('S4', 'C2'): 94.2, ('S4', 'C3'): 1776.34, ('S4', 'C4'): 285.48, ('S4', 'C5'): 868.85, ('S4', 'C6'): 86.55, ('S5', 'C1'): 1577, ('S5', 'C2'): 760.15, ('S5', 'C3'): 2090.19, ('S5', 'C4'): 43.2, ('S5', 'C5'): 1577.12, ('S5', 'C6'): 1095.17, ('S6', 'C1'): 49.14, ('S6', 'C2'): 4.33, ('S6', 'C3'): 2079.57, ('S6', 'C4'): 277.04, ('S6', 'C5'): 1032.01, ('S6', 'C6'): 1543.49}
    for s in suppliers:
        if s not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {s}')
    for c in stores:
        if c not in demand:
            raise ValueError(f'Missing demand for store {c}')
    for s in suppliers:
        for c in stores:
            if (s, c) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for supplier {s}, store {c}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(suppliers, stores, vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gp.quicksum((fixed_cost[s] * y[s] for s in suppliers)) + gp.quicksum((transportation_cost[s, c] * x[s, c] for s in suppliers for c in stores))
    m.setObjective(obj, GRB.MINIMIZE)
    for c in stores:
        m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name='demand_' + c)
    for s in suppliers:
        for c in stores:
            m.addConstr(x[s, c] <= demand[c] * y[s], name='link_' + s + '_' + c)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for s in suppliers:
            print(f'y[{s}] {y[s].VarName} {y[s].X}')
        for s in suppliers:
            for c in stores:
                print(f'x[{s},{c}] {x[s, c].VarName} {x[s, c].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_supplier_store_optimization()