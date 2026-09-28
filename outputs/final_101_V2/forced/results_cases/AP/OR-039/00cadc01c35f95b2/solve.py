import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
warehouses = ['Warehouse 1', 'Warehouse 2', 'Warehouse 3', 'Warehouse 4', 'Warehouse 5', 'Warehouse 6', 'Warehouse 7', 'Warehouse 8', 'Warehouse 9', 'Warehouse 10']
vehicle_value = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
vehicle_weight = {'Sedans': 20, 'SUVs': 15, 'Electric Vehicles': 25, 'Hybrid Vehicles': 18, 'Trucks': 10, 'Sports Cars': 5, 'Compact Cars': 22, 'Luxury Sedans': 8, 'Vans': 12, 'Pickup Trucks': 7}
warehouse_capacity = {'Warehouse 1': 100, 'Warehouse 2': 80, 'Warehouse 3': 120, 'Warehouse 4': 90, 'Warehouse 5': 50, 'Warehouse 6': 30, 'Warehouse 7': 110, 'Warehouse 8': 40, 'Warehouse 9': 60, 'Warehouse 10': 35}
if set(vehicle_types) != set(vehicle_value.keys()) or set(vehicle_types) != set(vehicle_weight.keys()):
    raise ValueError('Vehicle value/weight data missing for some vehicle types.')
if set(warehouses) != set(warehouse_capacity.keys()):
    raise ValueError('Warehouse capacity data missing for some warehouses.')
m = gp.Model('Car_Inventory_Warehouse')
x = m.addVars(vehicle_types, warehouses, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((vehicle_value[i] * x[i, w] for i in vehicle_types for w in warehouses)), GRB.MAXIMIZE)
for w in warehouses:
    m.addConstr(gp.quicksum((vehicle_weight[i] * x[i, w] for i in vehicle_types)) <= warehouse_capacity[w], name=f"cap_{w.replace(' ', '_')}")
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')