import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
stores = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
demand = {'C1': 45, 'C2': 23, 'C3': 94, 'C4': 92, 'C5': 57, 'C6': 52, 'C7': 23, 'C8': 99, 'C9': 99, 'C10': 77}
supply = {'S1': 127, 'S2': 236, 'S3': 168, 'S4': 115, 'S5': 280, 'S6': 179, 'S7': 135, 'S8': 263, 'S9': 283, 'S10': 476}
cost = {'S1': {'C1': 2077.05867, 'C2': 0.0, 'C3': 54.33526, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17285, 'C7': 0.0, 'C8': 0.0, 'C9': 169.33027, 'C10': 0.0}, 'S2': {'C1': 2077.05867, 'C2': 0.0, 'C3': 1141.04056, 'C4': 0.0, 'C5': 0.0, 'C6': 651.11123, 'C7': 0.0, 'C8': 0.0, 'C9': 8.06335, 'C10': 0.0}, 'S3': {'C1': 79.92103, 'C2': 474.24509, 'C3': 1477.06763, 'C4': 22.5831, 'C5': 474.24509, 'C6': 41.1066, 'C7': 474.24509, 'C8': 474.24509, 'C9': 624.16254, 'C10': 474.24509}, 'S4': {'C1': 1659.33693, 'C2': 57.20541, 'C3': 186.1519, 'C4': 1201.31371, 'C5': 1029.69746, 'C6': 41.82211, 'C7': 57.20541, 'C8': 1201.31371, 'C9': 884.56339, 'C10': 1029.69746}, 'S5': {'C1': 1297.2567, 'C2': 77.76629, 'C3': 24.2676, 'C4': 1399.79324, 'C5': 77.76629, 'C6': 53.91162, 'C7': 1399.79324, 'C8': 77.76629, 'C9': 1255.11515, 'C10': 1399.79324}, 'S6': {'C1': 1998.90907, 'C2': 985.31654, 'C3': 2.85417, 'C4': 1149.53597, 'C5': 985.31654, 'C6': 730.69236, 'C7': 54.73981, 'C8': 985.31654, 'C9': 46.8031, 'C10': 1149.53597}, 'S7': {'C1': 1780.33601, 'C2': 0.0, 'C3': 1141.04056, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17285, 'C7': 0.0, 'C8': 0.0, 'C9': 8.06335, 'C10': 0.0}, 'S8': {'C1': 75.40936, 'C2': 1338.19873, 'C3': 21.39135, 'C4': 74.34437, 'C5': 74.34437, 'C6': 937.35062, 'C7': 1338.19873, 'C8': 1338.19873, 'C9': 1392.11866, 'C10': 1338.19873}, 'S9': {'C1': 98.90756, 'C2': 0.0, 'C3': 978.03477, 'C4': 0.0, 'C5': 0.0, 'C6': 651.11123, 'C7': 0.0, 'C8': 0.0, 'C9': 169.33027, 'C10': 0.0}, 'S10': {'C1': 2077.05867, 'C2': 0.0, 'C3': 54.33526, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17285, 'C7': 0.0, 'C8': 0.0, 'C9': 145.14023, 'C10': 0.0}}
for i in warehouses:
    if i not in cost or i not in supply:
        raise ValueError(f'Missing cost or supply data for warehouse {i}')
    for j in stores:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for warehouse {i}, store {j}')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand data for store {j}')
m = gp.Model('transportation')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= supply[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')