import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
demand = {'C1': 45, 'C2': 23, 'C3': 94, 'C4': 92, 'C5': 57, 'C6': 52, 'C7': 23, 'C8': 99, 'C9': 99, 'C10': 77}
supply_capacity = {'S1': 127, 'S2': 236, 'S3': 168, 'S4': 115, 'S5': 280, 'S6': 179, 'S7': 135, 'S8': 263, 'S9': 283, 'S10': 476}
cost = {'S1': {'C1': 2077.06, 'C2': 0.0, 'C3': 54.34, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17, 'C7': 0.0, 'C8': 0.0, 'C9': 169.33, 'C10': 0.0}, 'S2': {'C1': 2077.06, 'C2': 0.0, 'C3': 1141.04, 'C4': 0.0, 'C5': 0.0, 'C6': 651.11, 'C7': 0.0, 'C8': 0.0, 'C9': 8.06, 'C10': 0.0}, 'S3': {'C1': 79.92, 'C2': 474.25, 'C3': 1477.07, 'C4': 22.58, 'C5': 474.25, 'C6': 41.11, 'C7': 474.25, 'C8': 474.25, 'C9': 624.16, 'C10': 474.25}, 'S4': {'C1': 1659.34, 'C2': 57.21, 'C3': 186.15, 'C4': 1201.31, 'C5': 1029.7, 'C6': 41.82, 'C7': 57.21, 'C8': 1201.31, 'C9': 884.56, 'C10': 1029.7}, 'S5': {'C1': 1297.26, 'C2': 77.77, 'C3': 24.27, 'C4': 1399.79, 'C5': 77.77, 'C6': 53.91, 'C7': 1399.79, 'C8': 77.77, 'C9': 1255.12, 'C10': 1399.79}, 'S6': {'C1': 1998.91, 'C2': 985.32, 'C3': 2.85, 'C4': 1149.54, 'C5': 985.32, 'C6': 730.69, 'C7': 54.74, 'C8': 985.32, 'C9': 46.8, 'C10': 1149.54}, 'S7': {'C1': 1780.34, 'C2': 0.0, 'C3': 1141.04, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17, 'C7': 0.0, 'C8': 0.0, 'C9': 8.06, 'C10': 0.0}, 'S8': {'C1': 75.41, 'C2': 1338.2, 'C3': 21.39, 'C4': 74.34, 'C5': 74.34, 'C6': 937.35, 'C7': 1338.2, 'C8': 1338.2, 'C9': 1392.12, 'C10': 1338.2}, 'S9': {'C1': 98.91, 'C2': 0.0, 'C3': 978.03, 'C4': 0.0, 'C5': 0.0, 'C6': 651.11, 'C7': 0.0, 'C8': 0.0, 'C9': 169.33, 'C10': 0.0}, 'S10': {'C1': 2077.06, 'C2': 0.0, 'C3': 54.34, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17, 'C7': 0.0, 'C8': 0.0, 'C9': 145.14, 'C10': 0.0}}
for i in warehouses:
    if i not in cost:
        raise ValueError(f'Missing cost data for warehouse {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for warehouse {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand data for customer {j}')
for i in warehouses:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for warehouse {i}')
m = gp.Model('transportation')
x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')