import gurobipy as gp
from gurobipy import GRB
suppliers = ['F1 (MOUNT AYR)', 'F2 (WAUKEE)', 'F3 (WAVERLY)', 'F4 (PELLA)', 'F5 (DES MOINES)']
stores = ['S1 (CLARINDA)', 'S2 (FORT MADISON)', 'S3 (SIOUX CITY)', 'S4 (TOLEDO)', 'S5 (BANCROFT)']
fixed_cost = {'F1 (MOUNT AYR)': 96.58, 'F2 (WAUKEE)': 94.06, 'F3 (WAVERLY)': 94.37, 'F4 (PELLA)': 82.88, 'F5 (DES MOINES)': 94.96}
cost = {'F1 (MOUNT AYR)': {'S1 (CLARINDA)': 694.68, 'S2 (FORT MADISON)': 17.48, 'S3 (SIOUX CITY)': 20.07, 'S4 (TOLEDO)': 199.02, 'S5 (BANCROFT)': 1685.53}, 'F2 (WAUKEE)': {'S1 (CLARINDA)': 15.13, 'S2 (FORT MADISON)': 1.5, 'S3 (SIOUX CITY)': 1.43, 'S4 (TOLEDO)': 27.88, 'S5 (BANCROFT)': 90.69}, 'F3 (WAVERLY)': {'S1 (CLARINDA)': 2.34, 'S2 (FORT MADISON)': 349.34, 'S3 (SIOUX CITY)': 246.6, 'S4 (TOLEDO)': 41.3, 'S5 (BANCROFT)': 78.73}, 'F4 (PELLA)': {'S1 (CLARINDA)': 1181.6, 'S2 (FORT MADISON)': 1458.53, 'S3 (SIOUX CITY)': 1646.36, 'S4 (TOLEDO)': 1924.55, 'S5 (BANCROFT)': 38.93}, 'F5 (DES MOINES)': {'S1 (CLARINDA)': 1030.8, 'S2 (FORT MADISON)': 43.48, 'S3 (SIOUX CITY)': 932.43, 'S4 (TOLEDO)': 55.39, 'S5 (BANCROFT)': 103.84}}
demand = {'S1 (CLARINDA)': 2397, 'S2 (FORT MADISON)': 1889, 'S3 (SIOUX CITY)': 2518, 'S4 (TOLEDO)': 3218, 'S5 (BANCROFT)': 1813}
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
m = gp.Model('Iowa_Liquor_Facility_Location')
x = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= M * y[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')