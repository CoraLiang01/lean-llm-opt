import gurobipy as gp
from gurobipy import GRB
facilities = ['Facility_1', 'Facility_2', 'Facility_3', 'Facility_4', 'Facility_5']
customers = ['Customer_1', 'Customer_2', 'Customer_3', 'Customer_4', 'Customer_5']
fixed_cost = {'Facility_1': 96.58, 'Facility_2': 94.06, 'Facility_3': 94.37, 'Facility_4': 82.88, 'Facility_5': 94.96}
demand = {'Customer_1': 2397, 'Customer_2': 1889, 'Customer_3': 2518, 'Customer_4': 3218, 'Customer_5': 1813}
cost = {'Facility_1': {'Customer_1': 694.68, 'Customer_2': 17.48, 'Customer_3': 20.07, 'Customer_4': 199.02, 'Customer_5': 1685.53}, 'Facility_2': {'Customer_1': 15.13, 'Customer_2': 1.5, 'Customer_3': 1.43, 'Customer_4': 27.88, 'Customer_5': 90.69}, 'Facility_3': {'Customer_1': 2.34, 'Customer_2': 349.34, 'Customer_3': 246.6, 'Customer_4': 41.3, 'Customer_5': 78.73}, 'Facility_4': {'Customer_1': 1181.6, 'Customer_2': 1458.53, 'Customer_3': 1646.36, 'Customer_4': 1924.55, 'Customer_5': 38.93}, 'Facility_5': {'Customer_1': 1030.8, 'Customer_2': 43.48, 'Customer_3': 932.43, 'Customer_4': 55.39, 'Customer_5': 103.84}}
M = sum(demand.values())
for i in facilities:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for {i}, {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for {j}')
m = gp.Model('Iowa_Liquor_FLP')
x = m.addVars(facilities, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(facilities, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in facilities for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in facilities)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in facilities)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in facilities), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')