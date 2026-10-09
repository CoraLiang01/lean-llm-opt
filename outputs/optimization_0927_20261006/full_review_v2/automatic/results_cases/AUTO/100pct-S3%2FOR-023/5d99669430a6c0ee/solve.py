import gurobipy as gp
from gurobipy import GRB
suppliers = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
stores = ['Customer_1', 'Customer_2', 'Customer_3', 'Customer_4', 'Customer_5']
demand = {'Customer_1': 2397, 'Customer_2': 1889, 'Customer_3': 2518, 'Customer_4': 3218, 'Customer_5': 1813}
fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
cost = {'MOUNT AYR': {'Customer_1': 694.68, 'Customer_2': 17.48, 'Customer_3': 20.07, 'Customer_4': 199.02, 'Customer_5': 1685.53}, 'WAUKEE': {'Customer_1': 15.13, 'Customer_2': 1.5, 'Customer_3': 1.43, 'Customer_4': 27.88, 'Customer_5': 90.69}, 'WAVERLY': {'Customer_1': 2.34, 'Customer_2': 349.34, 'Customer_3': 246.6, 'Customer_4': 41.3, 'Customer_5': 78.73}, 'PELLA': {'Customer_1': 1181.6, 'Customer_2': 1458.53, 'Customer_3': 1646.36, 'Customer_4': 1924.55, 'Customer_5': 38.93}, 'DES MOINES': {'Customer_1': 1030.8, 'Customer_2': 43.48, 'Customer_3': 932.43, 'Customer_4': 55.39, 'Customer_5': 103.84}}
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
x_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= M * y_vars[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')