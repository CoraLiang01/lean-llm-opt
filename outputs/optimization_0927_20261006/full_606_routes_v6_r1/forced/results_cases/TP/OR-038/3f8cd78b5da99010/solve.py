import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
benefit_coeff = {'Sedans': 1200, 'SUVs': 1800, 'Electric Vehicles': 2500, 'Hybrid Vehicles': 2000, 'Trucks': 1500, 'Sports Cars': 3000, 'Compact Cars': 1000, 'Luxury Sedans': 3500, 'Vans': 1600, 'Pickup Trucks': 1700}
inventory_limit = {'Sedans': 100, 'SUVs': 80, 'Electric Vehicles': 120, 'Hybrid Vehicles': 90, 'Trucks': 50, 'Sports Cars': 30, 'Compact Cars': 110, 'Luxury Sedans': 40, 'Vans': 60, 'Pickup Trucks': 35}
total_capacity = sum((inventory_limit[v] for v in vehicle_types))
if set(benefit_coeff.keys()) != set(vehicle_types):
    raise ValueError('Benefit coefficients missing for some vehicle types.')
if set(inventory_limit.keys()) != set(vehicle_types):
    raise ValueError('Inventory limits missing for some vehicle types.')
m = gp.Model('Norway_Car_Dealership_Inventory')
x_vars = m.addVars(vehicle_types, lb=0, ub={v: inventory_limit[v] for v in vehicle_types}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit_coeff[v] * x_vars[v] for v in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[v] for v in vehicle_types)) <= total_capacity, name='total_inventory')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in vehicle_types:
        print(f'{x_vars[v].VarName}: {x_vars[v].X}')
else:
    print(f'Solver status: {m.Status}')