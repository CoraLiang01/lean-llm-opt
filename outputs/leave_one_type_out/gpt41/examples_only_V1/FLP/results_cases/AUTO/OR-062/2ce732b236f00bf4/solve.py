import gurobipy as gp
from gurobipy import GRB

def solve_iowa_liquor_facility_location():
    suppliers = {1: 'MOUNT AYR', 2: 'WAUKEE', 3: 'WAVERLY', 4: 'PELLA', 5: 'DES MOINES'}
    stores = {1: 'CLARINDA', 2: 'FORT MADISON', 3: 'SIOUX CITY', 4: 'TOLEDO', 5: 'BANCROFT'}
    f = {1: 96.58, 2: 94.06, 3: 94.37, 4: 82.88, 5: 94.96}
    d = {1: 2397, 2: 1889, 3: 2518, 4: 3218, 5: 1813}
    c = {1: {1: 694.68, 2: 17.48, 3: 20.07, 4: 199.02, 5: 1685.53}, 2: {1: 15.13, 2: 1.5, 3: 1.43, 4: 27.88, 5: 90.69}, 3: {1: 2.34, 2: 349.34, 3: 246.6, 4: 41.3, 5: 78.73}, 4: {1: 1181.6, 2: 1458.53, 3: 1646.36, 4: 1924.55, 5: 38.93}, 5: {1: 1030.8, 2: 43.48, 3: 932.43, 4: 55.39, 5: 103.84}}
    if set(f.keys()) != set(suppliers.keys()):
        raise ValueError('Fixed cost data missing or extra for some suppliers.')
    if set(d.keys()) != set(stores.keys()):
        raise ValueError('Demand data missing or extra for some stores.')
    if set(c.keys()) != set(suppliers.keys()):
        raise ValueError('Transportation cost data missing or extra for some suppliers.')
    for i in suppliers:
        if set(c[i].keys()) != set(stores.keys()):
            raise ValueError(f'Transportation cost data missing or extra for supplier {i}.')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers.keys(), vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(suppliers.keys(), stores.keys(), vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in suppliers)) + gp.quicksum((c[i][j] * x[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
    for j in stores:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == d[j], name='demand')
    for i in suppliers:
        for j in stores:
            m.addConstr(x[i, j] <= d[j] * y[i], name='link')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_iowa_liquor_facility_location()