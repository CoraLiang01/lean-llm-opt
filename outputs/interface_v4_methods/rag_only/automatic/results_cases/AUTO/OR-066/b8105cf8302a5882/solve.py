import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    suppliers = ['S1', 'S2']
    supermarkets = ['C1', 'C2']
    fixed_cost = {'S1': 105.97, 'S2': 85.31}
    demand = {'C1': 144, 'C2': 216}
    transportation_cost = {('S1', 'C1'): 2358.39, ('S1', 'C2'): 1492.08, ('S2', 'C1'): 0.07, ('S2', 'C2'): 52.32}
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in supermarkets:
        if j not in demand:
            raise ValueError(f'Missing demand for supermarket {j}')
    for i in suppliers:
        for j in supermarkets:
            if (i, j) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({i},{j})')
    m = gp.Model('facility_location')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars(suppliers, supermarkets, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers))
    obj += gp.quicksum((transportation_cost[i, j] * x[i, j] for i in suppliers for j in supermarkets))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in supermarkets:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name='')
    total_demand = sum((demand[j] for j in supermarkets))
    for i in suppliers:
        m.addConstr(gp.quicksum((x[i, j] for j in supermarkets)) <= total_demand * y[i], name='')
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