import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    facilities = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
    customers = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
    fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
    demand = {'CLARINDA': 2397, 'FORT MADISON': 1889, 'SIOUX CITY': 2518, 'TOLEDO': 3218, 'BANCROFT': 1813}
    cost = {'MOUNT AYR': {'CLARINDA': 694.68, 'FORT MADISON': 17.48, 'SIOUX CITY': 20.07, 'TOLEDO': 199.02, 'BANCROFT': 1685.53}, 'WAUKEE': {'CLARINDA': 15.13, 'FORT MADISON': 1.5, 'SIOUX CITY': 1.43, 'TOLEDO': 27.88, 'BANCROFT': 90.69}, 'WAVERLY': {'CLARINDA': 2.34, 'FORT MADISON': 349.34, 'SIOUX CITY': 246.6, 'TOLEDO': 41.3, 'BANCROFT': 78.73}, 'PELLA': {'CLARINDA': 1181.6, 'FORT MADISON': 1458.53, 'SIOUX CITY': 1646.36, 'TOLEDO': 1924.55, 'BANCROFT': 38.93}, 'DES MOINES': {'CLARINDA': 1030.8, 'FORT MADISON': 43.48, 'SIOUX CITY': 932.43, 'TOLEDO': 55.39, 'BANCROFT': 103.84}}
    M = sum((demand[j] for j in customers))
    for i in facilities:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for facility {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for facility {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for facility {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('Iowa_Liquor_FLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(facilities, customers, lb=0, vtype=GRB.CONTINUOUS, name='x')
    y = m.addVars(facilities, vtype=GRB.BINARY, name='y')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in facilities for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in facilities)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in facilities)) == demand[j] for j in customers), name='demand')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in facilities), name='activation')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()