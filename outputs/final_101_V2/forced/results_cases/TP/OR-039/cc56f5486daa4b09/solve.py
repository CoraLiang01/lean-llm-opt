import gurobipy as gp
from gurobipy import GRB
warehouses = ['Warehouse 1', 'Warehouse 2', 'Warehouse 3', 'Warehouse 4', 'Warehouse 5', 'Warehouse 6', 'Warehouse 7', 'Warehouse 8', 'Warehouse 9', 'Warehouse 10']
products = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
capacity = {'Warehouse 1': 100, 'Warehouse 2': 80, 'Warehouse 3': 120, 'Warehouse 4': 90, 'Warehouse 5': 50, 'Warehouse 6': 30, 'Warehouse 7': 110, 'Warehouse 8': 40, 'Warehouse 9': 60, 'Warehouse 10': 35}
value = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
weight = {'Sedans': 20, 'SUVs': 15, 'Electric Vehicles': 25, 'Hybrid Vehicles': 18, 'Trucks': 10, 'Sports Cars': 5, 'Compact Cars': 22, 'Luxury Sedans': 8, 'Vans': 12, 'Pickup Trucks': 7}
for w in warehouses:
    if w not in capacity:
        raise ValueError(f'Missing capacity for warehouse {w}')
for p in products:
    if p not in value:
        raise ValueError(f'Missing value for product {p}')
    if p not in weight:
        raise ValueError(f'Missing weight for product {p}')
m = gp.Model('Norway_Car_Inventory')
x = m.addVars(warehouses, products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[p] * x[w, p] for w in warehouses for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x[w, p] for p in products)) <= capacity[w] for w in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')