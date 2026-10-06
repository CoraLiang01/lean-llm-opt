import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    suppliers = {1: 'MOUNT AYR', 2: 'WAUKEE', 3: 'WAVERLY', 4: 'PELLA', 5: 'DES MOINES'}
    stores = {1: 'Customer_1 (CLARINDA)', 2: 'Customer_2 (FORT MADISON)', 3: 'Customer_3 (SIOUX CITY)', 4: 'Customer_4 (TOLEDO)', 5: 'Customer_5 (BANCROFT)'}
    fixed_cost = {1: 96.58, 2: 94.06, 3: 94.37, 4: 82.88, 5: 94.96}
    demand = {1: 2397, 2: 1889, 3: 2518, 4: 3218, 5: 1813}
    transportation_cost = {1: {1: 694.68, 2: 17.48, 3: 20.07, 4: 199.02, 5: 1685.53}, 2: {1: 15.13, 2: 1.5, 3: 1.43, 4: 27.88, 5: 90.69}, 3: {1: 2.34, 2: 349.34, 3: 246.6, 4: 41.3, 5: 78.73}, 4: {1: 1181.6, 2: 1458.53, 3: 1646.36, 4: 1924.55, 5: 38.93}, 5: {1: 1030.8, 2: 43.48, 3: 932.43, 4: 55.39, 5: 103.84}}
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in transportation_cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in stores:
            if j not in transportation_cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, store {j}')
    for j in stores:
        if j not in demand:
            raise ValueError(f'Missing demand for store {j}')
    m = gp.Model('facility_location')
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers.keys(), vtype=GRB.BINARY, name='')
    x = m.addVars(suppliers.keys(), stores.keys(), vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_cost[i][j] * x[i, j] for i in suppliers for j in stores))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in stores:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name='')
    for i in suppliers:
        for j in stores:
            m.addConstr(x[i, j] <= demand[j] * y[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in suppliers:
            print(f'y[{i}] {y[i].X}')
        for i in suppliers:
            for j in stores:
                print(f'x[{i},{j}] {x[i, j].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()