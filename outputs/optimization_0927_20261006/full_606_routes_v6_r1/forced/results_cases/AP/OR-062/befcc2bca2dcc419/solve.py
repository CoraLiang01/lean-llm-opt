import gurobipy as gp
from gurobipy import GRB
suppliers = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
stores = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
demand_dict = {'Customer_1': 2397, 'Customer_2': 1889, 'Customer_3': 2518, 'Customer_4': 3218, 'Customer_5': 1813}
demand = dict(zip(stores, [demand_dict[f'Customer_{i + 1}'] for i in range(len(stores))]))
transport_cost = {'MOUNT AYR': {'CLARINDA': 694.68, 'FORT MADISON': 17.48, 'SIOUX CITY': 20.07, 'TOLEDO': 199.02, 'BANCROFT': 1685.53}, 'WAUKEE': {'CLARINDA': 15.13, 'FORT MADISON': 1.5, 'SIOUX CITY': 1.43, 'TOLEDO': 27.88, 'BANCROFT': 90.69}, 'WAVERLY': {'CLARINDA': 2.34, 'FORT MADISON': 349.34, 'SIOUX CITY': 246.6, 'TOLEDO': 41.3, 'BANCROFT': 78.73}, 'PELLA': {'CLARINDA': 1181.6, 'FORT MADISON': 1458.53, 'SIOUX CITY': 1646.36, 'TOLEDO': 1924.55, 'BANCROFT': 38.93}, 'DES MOINES': {'CLARINDA': 1030.8, 'FORT MADISON': 43.48, 'SIOUX CITY': 932.43, 'TOLEDO': 55.39, 'BANCROFT': 103.84}}
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed_cost for supplier {i}')
    if i not in transport_cost:
        raise ValueError(f'Missing transport_cost for supplier {i}')
    for j in stores:
        if j not in transport_cost[i]:
            raise ValueError(f'Missing transport_cost for supplier {i}, store {j}')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}')
m = gp.Model('Iowa_Liquor_Supplier_Selection')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(suppliers, stores, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)) + gp.quicksum((transport_cost[i][j] * x_vars[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
m.addConstrs((x_vars[i, j] <= demand[j] * y_vars[i] for i in suppliers for j in stores), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')