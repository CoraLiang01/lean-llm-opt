import gurobipy as gp
from gurobipy import GRB

def solve_iowa_liquor_supply():
    suppliers = {'MOUNT AYR': 0, 'WAUKEE': 1, 'WAVERLY': 2, 'PELLA': 3, 'DES MOINES': 4}
    stores = {'Customer_1 (CLARINDA)': 0, 'Customer_2 (FORT MADISON)': 1, 'Customer_3 (SIOUX CITY)': 2, 'Customer_4 (TOLEDO)': 3, 'Customer_5 (BANCROFT)': 4}
    supplier_list = list(suppliers.keys())
    store_list = list(stores.keys())
    fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
    demand = {'Customer_1 (CLARINDA)': 2397, 'Customer_2 (FORT MADISON)': 1889, 'Customer_3 (SIOUX CITY)': 2518, 'Customer_4 (TOLEDO)': 3218, 'Customer_5 (BANCROFT)': 1813}
    transportation_cost = {('MOUNT AYR', 'Customer_1 (CLARINDA)'): 694.68, ('MOUNT AYR', 'Customer_2 (FORT MADISON)'): 17.48, ('MOUNT AYR', 'Customer_3 (SIOUX CITY)'): 20.07, ('MOUNT AYR', 'Customer_4 (TOLEDO)'): 199.02, ('MOUNT AYR', 'Customer_5 (BANCROFT)'): 1685.53, ('WAUKEE', 'Customer_1 (CLARINDA)'): 15.13, ('WAUKEE', 'Customer_2 (FORT MADISON)'): 1.5, ('WAUKEE', 'Customer_3 (SIOUX CITY)'): 1.43, ('WAUKEE', 'Customer_4 (TOLEDO)'): 27.88, ('WAUKEE', 'Customer_5 (BANCROFT)'): 90.69, ('WAVERLY', 'Customer_1 (CLARINDA)'): 2.34, ('WAVERLY', 'Customer_2 (FORT MADISON)'): 349.34, ('WAVERLY', 'Customer_3 (SIOUX CITY)'): 246.6, ('WAVERLY', 'Customer_4 (TOLEDO)'): 41.3, ('WAVERLY', 'Customer_5 (BANCROFT)'): 78.73, ('PELLA', 'Customer_1 (CLARINDA)'): 1181.6, ('PELLA', 'Customer_2 (FORT MADISON)'): 1458.53, ('PELLA', 'Customer_3 (SIOUX CITY)'): 1646.36, ('PELLA', 'Customer_4 (TOLEDO)'): 1924.55, ('PELLA', 'Customer_5 (BANCROFT)'): 38.93, ('DES MOINES', 'Customer_1 (CLARINDA)'): 1030.8, ('DES MOINES', 'Customer_2 (FORT MADISON)'): 43.48, ('DES MOINES', 'Customer_3 (SIOUX CITY)'): 932.43, ('DES MOINES', 'Customer_4 (TOLEDO)'): 55.39, ('DES MOINES', 'Customer_5 (BANCROFT)'): 103.84}
    for s in supplier_list:
        if s not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {s}')
    for st in store_list:
        if st not in demand:
            raise ValueError(f'Missing demand for store {st}')
    for s in supplier_list:
        for st in store_list:
            if (s, st) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({s}, {st})')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(supplier_list, vtype=GRB.BINARY, name='')
    x = m.addVars(supplier_list, store_list, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((fixed_cost[s] * y[s] for s in supplier_list)) + gp.quicksum((transportation_cost[s, st] * x[s, st] for s in supplier_list for st in store_list)), GRB.MINIMIZE)
    for st in store_list:
        m.addConstr(gp.quicksum((x[s, st] for s in supplier_list)) == demand[st], name='')
    for s in supplier_list:
        for st in store_list:
            m.addConstr(x[s, st] <= demand[st] * y[s], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for s in supplier_list:
            print(f'y[{s}] {y[s].X}')
        for s in supplier_list:
            for st in store_list:
                print(f'x[{s},{st}] {x[s, st].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_iowa_liquor_supply()