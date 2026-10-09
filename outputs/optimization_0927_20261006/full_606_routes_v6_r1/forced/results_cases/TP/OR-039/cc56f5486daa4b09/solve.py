import gurobipy as gp
from gurobipy import GRB
products = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
product_values = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
product_weights = {'Sedans': 20, 'SUVs': 15, 'Electric Vehicles': 25, 'Hybrid Vehicles': 18, 'Trucks': 10, 'Sports Cars': 5, 'Compact Cars': 22, 'Luxury Sedans': 8, 'Vans': 12, 'Pickup Trucks': 7}
warehouses = ['Warehouse 1', 'Warehouse 2', 'Warehouse 3', 'Warehouse 4', 'Warehouse 5', 'Warehouse 6', 'Warehouse 7', 'Warehouse 8', 'Warehouse 9', 'Warehouse 10']
warehouse_capacities = {'Warehouse 1': 100, 'Warehouse 2': 80, 'Warehouse 3': 120, 'Warehouse 4': 90, 'Warehouse 5': 50, 'Warehouse 6': 30, 'Warehouse 7': 110, 'Warehouse 8': 40, 'Warehouse 9': 60, 'Warehouse 10': 35}
if set(product_values.keys()) != set(products):
    raise ValueError('Mismatch between products and product_values keys')
if set(product_weights.keys()) != set(products):
    raise ValueError('Mismatch between products and product_weights keys')
if set(warehouse_capacities.keys()) != set(warehouses):
    raise ValueError('Mismatch between warehouses and warehouse_capacities keys')
m = gp.Model('Car_Inventory_Optimization')
x_vars = m.addVars(products, warehouses, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[i] * x_vars[i, k] for i in products for k in warehouses)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_weights[i] * x_vars[i, k] for i in products)) <= warehouse_capacities[k] for k in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')