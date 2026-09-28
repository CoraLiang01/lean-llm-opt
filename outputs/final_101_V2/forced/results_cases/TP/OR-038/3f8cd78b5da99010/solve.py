import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
capacity = {'Sedans': 100, 'SUVs': 80, 'Electric Vehicles': 120, 'Hybrid Vehicles': 90, 'Trucks': 50, 'Sports Cars': 30, 'Compact Cars': 110, 'Luxury Sedans': 40, 'Vans': 60, 'Pickup Trucks': 35}
benefit = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
weight = {'Sedans': 20, 'SUVs': 15, 'Electric Vehicles': 25, 'Hybrid Vehicles': 18, 'Trucks': 10, 'Sports Cars': 5, 'Compact Cars': 22, 'Luxury Sedans': 8, 'Vans': 12, 'Pickup Trucks': 7}
for vt in vehicle_types:
    if vt not in capacity or vt not in benefit or vt not in weight:
        raise ValueError(f'Missing data for vehicle type: {vt}')
W = sum((capacity[vt] * weight[vt] for vt in vehicle_types))
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_types, lb=0, ub=[capacity[vt] for vt in vehicle_types], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vt] * x[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[vt] * x[vt] for vt in vehicle_types)) <= W, name='total_weight')
for vt in vehicle_types:
    m.addConstr(x[vt] >= 0, name=f'lb_{vt}')
    m.addConstr(x[vt] <= capacity[vt], name=f'ub_{vt}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for vt in vehicle_types:
        print(f'x[{vt}]: {x[vt].X}')
else:
    print(f'Solver status: {m.Status}')