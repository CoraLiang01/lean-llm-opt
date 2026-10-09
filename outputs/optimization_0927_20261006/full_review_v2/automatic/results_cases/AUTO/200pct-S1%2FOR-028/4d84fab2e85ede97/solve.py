import gurobipy as gp
from gurobipy import GRB
warehouses = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
stores = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
warehouse_opening_cost = {'1': 3000, '2': 3200, '3': 3100, '4': 2800, '5': 3500, '6': 2700, '7': 2900, '8': 3050, '9': 3100, '10': 2200, '11': 2890}
warehouse_capacity = {'1': 180, '2': 160, '3': 200, '4': 150, '5': 170, '6': 190, '7': 160, '8': 175, '9': 170, '10': 180, '11': 190}
store_demand = {'1': 30, '2': 40, '3': 20, '4': 35, '5': 20, '6': 25, '7': 45, '8': 38, '9': 32, '10': 41, '11': 44}
transportation_cost_matrix = [[12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15], [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16], [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17], [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18], [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17], [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19], [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14], [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18], [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18], [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19], [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]]
transportation_cost = {}
for (i_idx, i) in enumerate(warehouses):
    transportation_cost[i] = {}
    for (j_idx, j) in enumerate(stores):
        transportation_cost[i][j] = transportation_cost_matrix[i_idx][j_idx]
if set(warehouse_opening_cost.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse opening cost keys and warehouses list')
if set(warehouse_capacity.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse capacity keys and warehouses list')
if set(store_demand.keys()) != set(stores):
    raise ValueError('Mismatch in store demand keys and stores list')
for i in warehouses:
    if set(transportation_cost[i].keys()) != set(stores):
        raise ValueError(f'Mismatch in transportation cost keys for warehouse {i}')
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