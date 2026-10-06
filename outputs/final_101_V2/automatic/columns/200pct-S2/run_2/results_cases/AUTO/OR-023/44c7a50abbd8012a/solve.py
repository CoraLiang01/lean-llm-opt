import gurobipy as gp
from gurobipy import GRB
facilities = ['F1', 'F2', 'F3', 'F4', 'F5']
facility_names = {'F1': 'MOUNT AYR', 'F2': 'WAUKEE', 'F3': 'WAVERLY', 'F4': 'PELLA', 'F5': 'DES MOINES'}
customers = ['Customer_1', 'Customer_2', 'Customer_3', 'Customer_4', 'Customer_5']
demand = {'Customer_1': 2397, 'Customer_2': 1889, 'Customer_3': 2518, 'Customer_4': 3218, 'Customer_5': 1813}
fixed_cost = {'F1': 96.58, 'F2': 94.06, 'F3': 94.37, 'F4': 82.88, 'F5': 94.96}
cost = {'F1': {'Customer_1': 694.68, 'Customer_2': 17.48, 'Customer_3': 20.07, 'Customer_4': 199.02, 'Customer_5': 1685.53}, 'F2': {'Customer_1': 15.13, 'Customer_2': 1.5, 'Customer_3': 1.43, 'Customer_4': 27.88, 'Customer_5': 90.69}, 'F3': {'Customer_1': 2.34, 'Customer_2': 349.34, 'Customer_3': 246.6, 'Customer_4': 41.3, 'Customer_5': 78.73}, 'F4': {'Customer_1': 1181.6, 'Customer_2': 1458.53, 'Customer_3': 1646.36, 'Customer_4': 1924.55, 'Customer_5': 38.93}, 'F5': {'Customer_1': 1030.8, 'Customer_2': 43.48, 'Customer_3': 932.43, 'Customer_4': 55.39, 'Customer_5': 103.84}}
M = sum(demand.values())
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
m = gp.Model('Iowa_Liquor_Facility_Location')
x = m.addVars(facilities, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(facilities, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in facilities for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in facilities)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in facilities)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in facilities), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')