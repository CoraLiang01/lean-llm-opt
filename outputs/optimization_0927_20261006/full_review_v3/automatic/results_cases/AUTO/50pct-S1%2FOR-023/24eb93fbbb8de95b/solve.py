import gurobipy as gp
from gurobipy import GRB
suppliers = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
stores = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
demands = {'CLARINDA': 2397, 'FORT MADISON': 1889, 'SIOUX CITY': 2518, 'TOLEDO': 3218, 'BANCROFT': 1813}
fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
transport_cost = {'MOUNT AYR': {'CLARINDA': 694.68, 'FORT MADISON': 17.48, 'SIOUX CITY': 20.07, 'TOLEDO': 199.02, 'BANCROFT': 1685.53}, 'WAUKEE': {'CLARINDA': 15.13, 'FORT MADISON': 1.5, 'SIOUX CITY': 1.43, 'TOLEDO': 27.88, 'BANCROFT': 90.69}, 'WAVERLY': {'CLARINDA': 2.34, 'FORT MADISON': 349.34, 'SIOUX CITY': 246.6, 'TOLEDO': 41.3, 'BANCROFT': 78.73}, 'PELLA': {'CLARINDA': 1181.6, 'FORT MADISON': 1458.53, 'SIOUX CITY': 1646.36, 'TOLEDO': 1924.55, 'BANCROFT': 38.93}, 'DES MOINES': {'CLARINDA': 1030.8, 'FORT MADISON': 43.48, 'SIOUX CITY': 932.43, 'TOLEDO': 55.39, 'BANCROFT': 103.84}}
M = sum((demands[s] for s in stores))
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    if i not in transport_cost:
        raise ValueError(f'Missing transport cost row for supplier {i}')
    for s in stores:
        if s not in transport_cost[i]:
            raise ValueError(f'Missing transport cost for supplier {i}, store {s}')
for s in stores:
    if s not in demands:
        raise ValueError(f'Missing demand for store {s}')
m = gp.Model('Iowa_Liquor_Supplier_Selection')
x_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((transport_cost[i][s] * x_vars[i, s] for i in suppliers for s in stores)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, s] for i in suppliers)) == demands[s] for s in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, s] for s in stores)) <= M * y_vars[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')