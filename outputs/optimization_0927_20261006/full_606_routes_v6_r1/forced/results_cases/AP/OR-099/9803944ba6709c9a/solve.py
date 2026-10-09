import gurobipy as gp
from gurobipy import GRB
warehouses = [{'id': '1', 'opening_cost': 3000, 'capacity': 180}, {'id': '2', 'opening_cost': 3200, 'capacity': 160}, {'id': '3', 'opening_cost': 3100, 'capacity': 200}, {'id': '4', 'opening_cost': 2800, 'capacity': 150}, {'id': '5', 'opening_cost': 3500, 'capacity': 170}, {'id': '6', 'opening_cost': 2700, 'capacity': 190}, {'id': '7', 'opening_cost': 2900, 'capacity': 160}, {'id': '8', 'opening_cost': 3050, 'capacity': 175}, {'id': '9', 'opening_cost': 3100, 'capacity': 170}, {'id': '10', 'opening_cost': 2200, 'capacity': 180}, {'id': '11', 'opening_cost': 2890, 'capacity': 190}]
stores = [{'id': '1', 'demand': 30}, {'id': '2', 'demand': 40}, {'id': '3', 'demand': 20}, {'id': '4', 'demand': 35}, {'id': '5', 'demand': 20}, {'id': '6', 'demand': 25}, {'id': '7', 'demand': 45}, {'id': '8', 'demand': 38}, {'id': '9', 'demand': 32}, {'id': '10', 'demand': 41}, {'id': '11', 'demand': 44}]
transportation_cost = {'1': {'1': 12, '2': 11, '3': 14, '4': 15, '5': 17, '6': 13, '7': 12, '8': 16, '9': 16, '10': 14, '11': 15}, '2': {'1': 17, '2': 19, '3': 15, '4': 20, '5': 18, '6': 14, '7': 17, '8': 15, '9': 13, '10': 15, '11': 16}, '3': {'1': 13, '2': 14, '3': 12, '4': 14, '5': 16, '6': 15, '7': 11, '8': 14, '9': 16, '10': 18, '11': 17}, '4': {'1': 18, '2': 16, '3': 17, '4': 13, '5': 18, '6': 17, '7': 14, '8': 19, '9': 16, '10': 13, '11': 18}, '5': {'1': 10, '2': 13, '3': 12, '4': 19, '5': 15, '6': 11, '7': 12, '8': 14, '9': 12, '10': 15, '11': 17}, '6': {'1': 15, '2': 12, '3': 14, '4': 16, '5': 13, '6': 17, '7': 16, '8': 16, '9': 14, '10': 18, '11': 19}, '7': {'1': 14, '2': 13, '3': 15, '4': 17, '5': 12, '6': 13, '7': 14, '8': 15, '9': 12, '10': 16, '11': 14}, '8': {'1': 19, '2': 16, '3': 18, '4': 20, '5': 17, '6': 19, '7': 16, '8': 18, '9': 15, '10': 15, '11': 18}, '9': {'1': 17, '2': 18, '3': 12, '4': 14, '5': 16, '6': 15, '7': 14, '8': 17, '9': 21, '10': 15, '11': 18}, '10': {'1': 14, '2': 13, '3': 15, '4': 17, '5': 16, '6': 18, '7': 14, '8': 19, '9': 15, '10': 17, '11': 19}, '11': {'1': 15, '2': 13, '3': 16, '4': 17, '5': 11, '6': 13, '7': 14, '8': 15, '9': 19, '10': 21, '11': 13}}
warehouse_ids = [w['id'] for w in warehouses]
store_ids = [s['id'] for s in stores]
f = {w['id']: w['opening_cost'] for w in warehouses}
cap = {w['id']: w['capacity'] for w in warehouses}
d = {s['id']: s['demand'] for s in stores}
c = {wi: {sj: transportation_cost[wi][sj] for sj in store_ids} for wi in warehouse_ids}
for wi in warehouse_ids:
    if wi not in c or len(c[wi]) != len(store_ids):
        raise ValueError(f'Missing transportation cost data for warehouse {wi}')
for sj in store_ids:
    if sj not in d:
        raise ValueError(f'Missing demand data for store {sj}')
for wi in warehouse_ids:
    if wi not in f or wi not in cap:
        raise ValueError(f'Missing opening cost or capacity for warehouse {wi}')
m = gp.Model('Warehouse_Location')
y_vars = m.addVars(warehouse_ids, vtype=GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, store_ids, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((f[wi] * y_vars[wi] for wi in warehouse_ids)) + gp.quicksum((c[wi][sj] * x_vars[wi, sj] for wi in warehouse_ids for sj in store_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wi, sj] for wi in warehouse_ids)) == d[sj] for sj in store_ids), name='')
m.addConstrs((gp.quicksum((x_vars[wi, sj] for sj in store_ids)) <= cap[wi] * y_vars[wi] for wi in warehouse_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')