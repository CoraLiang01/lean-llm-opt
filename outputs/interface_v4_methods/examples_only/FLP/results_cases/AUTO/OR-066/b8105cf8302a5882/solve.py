import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    suppliers = ['S1', 'S2']
    supermarkets = ['C1', 'C2']
    fixed_cost = {'S1': 105.97, 'S2': 85.31}
    transportation_cost = {('S1', 'C1'): 2358.39, ('S1', 'C2'): 1492.08, ('S2', 'C1'): 0.07, ('S2', 'C2'): 52.32}
    demand = {'C1': 144, 'C2': 216}
    M = sum(demand.values())
    for s in suppliers:
        if s not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {s}')
    for s in suppliers:
        for c in supermarkets:
            if (s, c) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({s},{c})')
    for c in supermarkets:
        if c not in demand:
            raise ValueError(f'Missing demand for supermarket {c}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars(suppliers, supermarkets, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.quicksum((fixed_cost[s] * y[s] for s in suppliers)) + gp.quicksum((transportation_cost[s, c] * x[s, c] for s in suppliers for c in supermarkets))
    m.setObjective(obj, GRB.MINIMIZE)
    for c in supermarkets:
        m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name='')
    for s in suppliers:
        m.addConstr(gp.quicksum((x[s, c] for c in supermarkets)) <= M * y[s], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()