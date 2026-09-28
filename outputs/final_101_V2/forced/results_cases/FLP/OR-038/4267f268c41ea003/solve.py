import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
benefit = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
capacity = {'Sedans': 100, 'SUVs': 80, 'Electric Vehicles': 120, 'Hybrid Vehicles': 90, 'Trucks': 50, 'Sports Cars': 30, 'Compact Cars': 110, 'Luxury Sedans': 40, 'Vans': 60, 'Pickup Trucks': 35}
if set(vehicle_types) != set(benefit.keys()):
    raise ValueError('Benefit coefficients missing for some vehicle types.')
if set(vehicle_types) != set(capacity.keys()):
    raise ValueError('Capacity limits missing for some vehicle types.')
m = gp.Model('Car_Inventory_Replenishment')
x = m.addVars(vehicle_types, lb=0, ub=[capacity[i] for i in vehicle_types], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[i] * x[i] for i in vehicle_types)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= capacity[i] for i in vehicle_types), name='')
m.addConstrs((x[i] >= 0 for i in vehicle_types), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')