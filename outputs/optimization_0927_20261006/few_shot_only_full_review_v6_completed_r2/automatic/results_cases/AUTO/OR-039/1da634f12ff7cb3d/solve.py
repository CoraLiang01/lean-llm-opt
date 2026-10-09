import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
warehouses = ['Warehouse 1', 'Warehouse 2', 'Warehouse 3', 'Warehouse 4', 'Warehouse 5', 'Warehouse 6', 'Warehouse 7', 'Warehouse 8', 'Warehouse 9', 'Warehouse 10']
value = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
weight = {'Sedans': 20, 'SUVs': 15, 'Electric Vehicles': 25, 'Hybrid Vehicles': 18, 'Trucks': 10, 'Sports Cars': 5, 'Compact Cars': 22, 'Luxury Sedans': 8, 'Vans': 12, 'Pickup Trucks': 7}
capacity = {'Warehouse 1': 100, 'Warehouse 2': 80, 'Warehouse 3': 120, 'Warehouse 4': 90, 'Warehouse 5': 50, 'Warehouse 6': 30, 'Warehouse 7': 110, 'Warehouse 8': 40, 'Warehouse 9': 60, 'Warehouse 10': 35}
for i in vehicle_types:
    if i not in value or i not in weight:
        raise ValueError(f'Missing value or weight for vehicle type {i}')
for j in warehouses:
    if j not in capacity:
        raise ValueError(f'Missing capacity for warehouse {j}')
m = gp.Model('Car_Dealership_Inventory')
x_vars = m.addVars(vehicle_types, warehouses, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[i] * x_vars[i, j] for i in vehicle_types for j in warehouses)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[i] * x_vars[i, j] for i in vehicle_types)) <= capacity[j] for j in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')