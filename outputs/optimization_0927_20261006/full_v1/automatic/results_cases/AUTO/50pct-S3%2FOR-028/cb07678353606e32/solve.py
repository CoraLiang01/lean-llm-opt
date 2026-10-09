import gurobipy as gp
from gurobipy import GRB
warehouses = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
stores = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
warehouse_opening_cost = {'1': 3000, '2': 3200, '3': 3100, '4': 2800, '5': 3500, '6': 2700, '7': 2900, '8': 3050, '9': 3100, '10': 2200, '11': 2890}
warehouse_capacity = {'1': 180, '2': 160, '3': 200, '4': 150, '5': 170, '6': 190, '7': 160, '8': 175, '9': 170, '10': 180, '11': 190}
store_demand = {'1': 30, '2': 40, '3': 20, '4': 35, '5': 20, '6': 25, '7': 45, '8': 38, '9': 32, '10': 41, '11': 44}
transportation_cost = {'1': {'1': 12, '2': 17, '3': 13, '4': 18, '5': 10, '6': 15, '7': 14, '8': 19, '9': 17, '10': 14, '11': 15}, '2': {'1': 11, '2': 19, '3': 14, '4': 16, '5': 13, '6': 12, '7': 13, '8': 16, '9': 18, '10': 13, '11': 13}, '3': {'1': 14, '2': 15, '3': 12, '4': 17, '5': 12, '6': 14, '7': 15, '8': 18, '9': 12, '10': 15, '11': 16}, '4': {'1': 15, '2': 20, '3': 14, '4': 13, '5': 19, '6': 16, '7': 17, '8': 20, '9': 14, '10': 17, '11': 17}, '5': {'1': 17, '2': 18, '3': 16, '4': 18, '5': 15, '6': 13, '7': 12, '8': 17, '9': 16, '10': 16, '11': 11}, '6': {'1': 13, '2': 14, '3': 15, '4': 17, '5': 11, '6': 17, '7': 13, '8': 19, '9': 15, '10': 18, '11': 13}, '7': {'1': 12, '2': 17, '3': 11, '4': 14, '5': 12, '6': 16, '7': 14, '8': 16, '9': 14, '10': 14, '11': 14}, '8': {'1': 16, '2': 15, '3': 14, '4': 19, '5': 14, '6': 16, '7': 15, '8': 18, '9': 17, '10': 19, '11': 15}, '9': {'1': 16, '2': 13, '3': 16, '4': 16, '5': 12, '6': 14, '7': 12, '8': 15, '9': 21, '10': 15, '11': 19}, '10': {'1': 14, '2': 15, '3': 18, '4': 13, '5': 15, '6': 18, '7': 16, '8': 15, '9': 15, '10': 17, '11': 21}, '11': {'1': 15, '2': 16, '3': 17, '4': 18, '5': 17, '6': 19, '7': 14, '8': 18, '9': 18, '10': 19, '11': 13}}
for i in warehouses:
    if i not in warehouse_opening_cost or i not in warehouse_capacity:
        raise ValueError(f'Missing warehouse data for {i}')
    if i not in transportation_cost:
        raise ValueError(f'Missing transportation cost row for warehouse {i}')
    for j in stores:
        if j not in transportation_cost[i]:
            raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')
for j in stores:
    if j not in store_demand:
        raise ValueError(f'Missing demand for store {j}')
m = gp.Model('Warehouse_Location')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((transportation_cost[i][j] * x_vars[i, j] for i in warehouses for j in stores)) + gp.quicksum((warehouse_opening_cost[i] * y_vars[i] for i in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == store_demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= warehouse_capacity[i] * y_vars[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')