import gurobipy as gp
from gurobipy import GRB
warehouses = {'1': {'opening_cost': 3000, 'capacity': 180}, '2': {'opening_cost': 3200, 'capacity': 160}, '3': {'opening_cost': 3100, 'capacity': 200}, '4': {'opening_cost': 2800, 'capacity': 150}, '5': {'opening_cost': 3500, 'capacity': 170}, '6': {'opening_cost': 2700, 'capacity': 190}, '7': {'opening_cost': 2900, 'capacity': 160}, '8': {'opening_cost': 3050, 'capacity': 175}, '9': {'opening_cost': 3100, 'capacity': 170}, '10': {'opening_cost': 2200, 'capacity': 180}, '11': {'opening_cost': 2890, 'capacity': 190}}
stores = {'1': 30, '2': 40, '3': 20, '4': 35, '5': 20, '6': 25, '7': 45, '8': 38, '9': 32, '10': 41, '11': 44}
transportation_cost = {'1': {'1': 12, '2': 11, '3': 14, '4': 15, '5': 17, '6': 13, '7': 12, '8': 16, '9': 16, '10': 14, '11': 15}, '2': {'1': 17, '2': 19, '3': 15, '4': 20, '5': 18, '6': 14, '7': 17, '8': 15, '9': 13, '10': 15, '11': 16}, '3': {'1': 13, '2': 14, '3': 12, '4': 14, '5': 16, '6': 15, '7': 11, '8': 14, '9': 16, '10': 18, '11': 17}, '4': {'1': 18, '2': 16, '3': 17, '4': 13, '5': 18, '6': 17, '7': 14, '8': 19, '9': 16, '10': 13, '11': 18}, '5': {'1': 10, '2': 13, '3': 12, '4': 19, '5': 15, '6': 11, '7': 12, '8': 14, '9': 12, '10': 15, '11': 17}, '6': {'1': 15, '2': 12, '3': 14, '4': 16, '5': 13, '6': 17, '7': 16, '8': 16, '9': 14, '10': 18, '11': 19}, '7': {'1': 14, '2': 13, '3': 15, '4': 17, '5': 12, '6': 13, '7': 14, '8': 15, '9': 12, '10': 16, '11': 14}, '8': {'1': 19, '2': 16, '3': 18, '4': 20, '5': 17, '6': 19, '7': 16, '8': 18, '9': 15, '10': 15, '11': 18}, '9': {'1': 17, '2': 18, '3': 12, '4': 14, '5': 16, '6': 15, '7': 14, '8': 17, '9': 21, '10': 15, '11': 18}, '10': {'1': 14, '2': 13, '3': 15, '4': 17, '5': 16, '6': 18, '7': 14, '8': 19, '9': 15, '10': 17, '11': 19}, '11': {'1': 15, '2': 13, '3': 16, '4': 17, '5': 11, '6': 13, '7': 14, '8': 15, '9': 19, '10': 21, '11': 13}}
warehouse_ids = list(warehouses.keys())
store_ids = list(stores.keys())
for i in warehouse_ids:
    if i not in transportation_cost:
        raise ValueError(f'Missing transportation_cost for warehouse {i}')
    for j in store_ids:
        if j not in transportation_cost[i]:
            raise ValueError(f'Missing transportation_cost for warehouse {i}, store {j}')
m = gp.Model('Warehouse_Location')
y = m.addVars(warehouse_ids, vtype=GRB.BINARY, name='')
x = m.addVars(warehouse_ids, store_ids, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((warehouses[i]['opening_cost'] * y[i] for i in warehouse_ids)) + gp.quicksum((transportation_cost[i][j] * x[i, j] for i in warehouse_ids for j in store_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in warehouse_ids)) == stores[j] for j in store_ids), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in store_ids)) <= warehouses[i]['capacity'] * y[i] for i in warehouse_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')