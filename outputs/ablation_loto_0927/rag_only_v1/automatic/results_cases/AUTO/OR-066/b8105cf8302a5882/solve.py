import gurobipy as gp
from gurobipy import GRB

def solve_facility_location():
    facilities = ['S1', 'S2']
    customers = ['C1', 'C2']
    fixed_cost = {'S1': 105.97, 'S2': 85.31}
    transportation_cost = {('S1', 'C1'): 2358.39, ('S1', 'C2'): 1492.08, ('S2', 'C1'): 0.07, ('S2', 'C2'): 52.32}
    demand = {'C1': 144, 'C2': 216}
    for i in facilities:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for facility {i}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in facilities:
        for j in customers:
            if (i, j) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({i},{j})')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(facilities, vtype=GRB.BINARY, name='')
    x = m.addVars(facilities, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((transportation_cost[i, j] * x[i, j] for i in facilities for j in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j], name='')
    for i in facilities:
        for j in customers:
            m.addConstr(x[i, j] <= demand[j] * y[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_facility_location()