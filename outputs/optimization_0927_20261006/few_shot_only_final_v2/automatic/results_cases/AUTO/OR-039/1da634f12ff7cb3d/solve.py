import gurobipy as gp
from gurobipy import GRB
products = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
warehouses = ['Warehouse 1', 'Warehouse 2', 'Warehouse 3', 'Warehouse 4', 'Warehouse 5', 'Warehouse 6', 'Warehouse 7', 'Warehouse 8', 'Warehouse 9', 'Warehouse 10']
value = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
weight = {'Sedans': 20, 'SUVs': 15, 'Electric Vehicles': 25, 'Hybrid Vehicles': 18, 'Trucks': 10, 'Sports Cars': 5, 'Compact Cars': 22, 'Luxury Sedans': 8, 'Vans': 12, 'Pickup Trucks': 7}
capacity = {'Warehouse 1': 100, 'Warehouse 2': 80, 'Warehouse 3': 120, 'Warehouse 4': 90, 'Warehouse 5': 50, 'Warehouse 6': 30, 'Warehouse 7': 110, 'Warehouse 8': 40, 'Warehouse 9': 60, 'Warehouse 10': 35}
if set(value.keys()) != set(products):
    raise ValueError('Value data missing for some products.')
if set(weight.keys()) != set(products):
    raise ValueError('Weight data missing for some products.')
if set(capacity.keys()) != set(warehouses):
    raise ValueError('Capacity data missing for some warehouses.')
m = gp.Model('Car_Inventory_Allocation')
x_vars = m.addVars(products, warehouses, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[p, w] for p in products for w in warehouses)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x_vars[p, w] for p in products)) <= capacity[w] for w in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')