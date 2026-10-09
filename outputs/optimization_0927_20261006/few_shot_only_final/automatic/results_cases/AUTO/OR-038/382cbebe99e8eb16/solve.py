import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedans', 'SUVs', 'Electric Vehicles', 'Hybrid Vehicles', 'Trucks', 'Sports Cars', 'Compact Cars', 'Luxury Sedans', 'Vans', 'Pickup Trucks']
vehicle_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
capacity = {1: 100, 2: 80, 3: 120, 4: 90, 5: 50, 6: 30, 7: 110, 8: 40, 9: 60, 10: 35}
value = {1: 1200, 2: 1800, 3: 2500, 4: 2000, 5: 1500, 6: 3000, 7: 1000, 8: 3500, 9: 1600, 10: 1700}
if set(capacity.keys()) != set(vehicle_ids) or set(value.keys()) != set(vehicle_ids):
    raise ValueError('Missing data for some vehicle types.')
m = gp.Model('Car_Inventory_Optimization')
x_vars = m.addVars(vehicle_ids, lb=0, ub=[capacity[i] for i in vehicle_ids], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[i] * x_vars[i] for i in vehicle_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[i] for i in vehicle_ids)) <= sum((capacity[i] for i in vehicle_ids)), name='total_inventory')
m.addConstrs((x_vars[i] <= capacity[i] for i in vehicle_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')