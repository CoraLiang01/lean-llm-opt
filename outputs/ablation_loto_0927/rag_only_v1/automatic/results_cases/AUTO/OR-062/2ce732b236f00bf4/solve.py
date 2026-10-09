import gurobipy as gp
from gurobipy import GRB

def solve_iowa_liquor_optimization():
    suppliers = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
    stores = ['Customer_1 (CLARINDA)', 'Customer_2 (FORT MADISON)', 'Customer_3 (SIOUX CITY)', 'Customer_4 (TOLEDO)', 'Customer_5 (BANCROFT)']
    fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
    demand = {'Customer_1 (CLARINDA)': 2397, 'Customer_2 (FORT MADISON)': 1889, 'Customer_3 (SIOUX CITY)': 2518, 'Customer_4 (TOLEDO)': 3218, 'Customer_5 (BANCROFT)': 1813}
    transportation_cost = {'MOUNT AYR': {'Customer_1 (CLARINDA)': 694.68, 'Customer_2 (FORT MADISON)': 17.48, 'Customer_3 (SIOUX CITY)': 20.07, 'Customer_4 (TOLEDO)': 199.02, 'Customer_5 (BANCROFT)': 1685.53}, 'WAUKEE': {'Customer_1 (CLARINDA)': 15.13, 'Customer_2 (FORT MADISON)': 1.5, 'Customer_3 (SIOUX CITY)': 1.43, 'Customer_4 (TOLEDO)': 27.88, 'Customer_5 (BANCROFT)': 90.69}, 'WAVERLY': {'Customer_1 (CLARINDA)': 2.34, 'Customer_2 (FORT MADISON)': 349.34, 'Customer_3 (SIOUX CITY)': 246.6, 'Customer_4 (TOLEDO)': 41.3, 'Customer_5 (BANCROFT)': 78.73}, 'PELLA': {'Customer_1 (CLARINDA)': 1181.6, 'Customer_2 (FORT MADISON)': 1458.53, 'Customer_3 (SIOUX CITY)': 1646.36, 'Customer_4 (TOLEDO)': 1924.55, 'Customer_5 (BANCROFT)': 38.93}, 'DES MOINES': {'Customer_1 (CLARINDA)': 1030.8, 'Customer_2 (FORT MADISON)': 43.48, 'Customer_3 (SIOUX CITY)': 932.43, 'Customer_4 (TOLEDO)': 55.39, 'Customer_5 (BANCROFT)': 103.84}}
    if set(fixed_cost.keys()) != set(suppliers):
        raise ValueError('Fixed cost data missing or extra suppliers.')
    if set(demand.keys()) != set(stores):
        raise ValueError('Demand data missing or extra stores.')
    if set(transportation_cost.keys()) != set(suppliers):
        raise ValueError('Transportation cost data missing or extra suppliers.')
    for i in suppliers:
        if set(transportation_cost[i].keys()) != set(stores):
            raise ValueError(f'Transportation cost data missing or extra stores for supplier {i}.')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars(suppliers, stores, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_cost[i][j] * x[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
    for j in stores:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name='')
    for i in suppliers:
        for j in stores:
            m.addConstr(x[i, j] <= demand[j] * y[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_iowa_liquor_optimization()