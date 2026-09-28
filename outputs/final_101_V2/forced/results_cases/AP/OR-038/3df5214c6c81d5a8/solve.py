import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
benefit_coefficients = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
daily_inventory_limits = {'Sedans': 100, 'SUVs': 80, 'Electric Vehicles': 120, 'Hybrid Vehicles': 90, 'Trucks': 50, 'Sports Cars': 30, 'Compact Cars': 110, 'Luxury Sedans': 40, 'Vans': 60, 'Pickup Trucks': 35}
T = 400
if set(vehicle_types) != set(benefit_coefficients.keys()):
    raise ValueError('Mismatch between vehicle_types and benefit_coefficients keys')
if set(vehicle_types) != set(daily_inventory_limits.keys()):
    raise ValueError('Mismatch between vehicle_types and daily_inventory_limits keys')
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_types, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((benefit_coefficients[v] * x[v] for v in vehicle_types)), GRB.MAXIMIZE)
for v in vehicle_types:
    m.addConstr(x[v] <= daily_inventory_limits[v], name=f'cap_{v}')
m.addConstr(gp.quicksum((x[v] for v in vehicle_types)) <= T, name='total_inventory')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')