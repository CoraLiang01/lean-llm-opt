import gurobipy as gp
from gurobipy import GRB
suppliers = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
stores = ['SIOUX CITY', 'CLARINDA', 'FORT MADISON', 'TOLEDO', 'BANCROFT']
fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
demand = {'SIOUX CITY': 2397, 'CLARINDA': 1889, 'FORT MADISON': 2518, 'TOLEDO': 3218, 'BANCROFT': 1813}
cost = {'MOUNT AYR': {'SIOUX CITY': 20.07, 'CLARINDA': 694.68, 'FORT MADISON': 17.48, 'TOLEDO': 199.02, 'BANCROFT': 1685.53}, 'WAUKEE': {'SIOUX CITY': 1.43, 'CLARINDA': 15.13, 'FORT MADISON': 1.5, 'TOLEDO': 27.88, 'BANCROFT': 90.69}, 'WAVERLY': {'SIOUX CITY': 246.6, 'CLARINDA': 2.34, 'FORT MADISON': 349.34, 'TOLEDO': 41.3, 'BANCROFT': 78.73}, 'PELLA': {'SIOUX CITY': 1646.36, 'CLARINDA': 1181.6, 'FORT MADISON': 1458.53, 'TOLEDO': 1924.55, 'BANCROFT': 38.93}, 'DES MOINES': {'SIOUX CITY': 932.43, 'CLARINDA': 1030.8, 'FORT MADISON': 43.48, 'TOLEDO': 55.39, 'BANCROFT': 103.84}}
M = sum(demand.values())
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for supplier {i}')
    for j in stores:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for supplier {i}, store {j}')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}')
m = gp.Model('Iowa_Liquor_FLP')
x = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= M * y[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')