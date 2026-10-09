import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
value = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
capacity = {'Sedans': 100, 'SUVs': 80, 'Electric Vehicles': 120, 'Hybrid Vehicles': 90, 'Trucks': 50, 'Sports Cars': 30, 'Compact Cars': 110, 'Luxury Sedans': 40, 'Vans': 60, 'Pickup Trucks': 35}
if set(vehicle_types) != set(value) or set(vehicle_types) != set(capacity):
    raise ValueError('Mismatch in vehicle_types, value, or capacity keys.')
m = gp.Model('Car_Dealership_Inventory')
x_vars = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[i] * x_vars[i] for i in vehicle_types)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= capacity[i] for i in vehicle_types), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')