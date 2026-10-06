import gurobipy as gp
from gurobipy import GRB

def solve_adidas_supplier_store_allocation():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    stores = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    fixed_cost = {'S1': 98.88, 'S2': 99.73, 'S3': 94.01, 'S4': 93.77, 'S5': 107.59, 'S6': 112.65}
    transportation_cost = {('S1', 'C1'): 4.2, ('S1', 'C2'): 5.1, ('S1', 'C3'): 3.8, ('S1', 'C4'): 6.0, ('S1', 'C5'): 4.7, ('S1', 'C6'): 5.5, ('S2', 'C1'): 5.0, ('S2', 'C2'): 4.8, ('S2', 'C3'): 4.5, ('S2', 'C4'): 5.9, ('S2', 'C5'): 5.2, ('S2', 'C6'): 6.1, ('S3', 'C1'): 4.7, ('S3', 'C2'): 5.3, ('S3', 'C3'): 4.0, ('S3', 'C4'): 5.7, ('S3', 'C5'): 4.9, ('S3', 'C6'): 5.8, ('S4', 'C1'): 5.3, ('S4', 'C2'): 5.0, ('S4', 'C3'): 4.6, ('S4', 'C4'): 6.2, ('S4', 'C5'): 5.1, ('S4', 'C6'): 6.0, ('S5', 'C1'): 4.9, ('S5', 'C2'): 5.2, ('S5', 'C3'): 4.3, ('S5', 'C4'): 5.8, ('S5', 'C5'): 5.0, ('S5', 'C6'): 5.9, ('S6', 'C1'): 5.1, ('S6', 'C2'): 5.4, ('S6', 'C3'): 4.4, ('S6', 'C4'): 6.1, ('S6', 'C5'): 5.3, ('S6', 'C6'): 6.2}
    demand = {'C1': 120, 'C2': 150, 'C3': 100, 'C4': 130, 'C5': 110, 'C6': 140}
    for s in suppliers:
        if s not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {s}')
        for c in stores:
            if (s, c) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for supplier {s}, store {c}')
    for c in stores:
        if c not in demand:
            raise ValueError(f'Missing demand for store {c}')
    M = sum((demand[c] for c in stores))
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars(suppliers, stores, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((fixed_cost[s] * y[s] for s in suppliers)) + gp.quicksum((transportation_cost[s, c] * x[s, c] for s in suppliers for c in stores)), GRB.MINIMIZE)
    for c in stores:
        m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name='')
    for s in suppliers:
        m.addConstr(gp.quicksum((x[s, c] for c in stores)) <= M * y[s], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for s in suppliers:
            print(f'y[{s}] {y[s].X}')
        for s in suppliers:
            for c in stores:
                print(f'x[{s},{c}] {x[s, c].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_adidas_supplier_store_allocation()