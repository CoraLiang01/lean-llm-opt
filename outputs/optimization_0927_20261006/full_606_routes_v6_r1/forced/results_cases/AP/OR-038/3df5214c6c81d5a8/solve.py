import gurobipy as gp
from gurobipy import GRB
vehicle_types = [{'VehicleID': 1, 'VehicleType': 'Sedans', 'Capacity': 100, 'Value': 1200}, {'VehicleID': 2, 'VehicleType': 'SUVs', 'Capacity': 80, 'Value': 1800}, {'VehicleID': 3, 'VehicleType': 'Electric Vehicles', 'Capacity': 120, 'Value': 2500}, {'VehicleID': 4, 'VehicleType': 'Hybrid Vehicles', 'Capacity': 90, 'Value': 2000}, {'VehicleID': 5, 'VehicleType': 'Trucks', 'Capacity': 50, 'Value': 1500}, {'VehicleID': 6, 'VehicleType': 'Sports Cars', 'Capacity': 30, 'Value': 3000}, {'VehicleID': 7, 'VehicleType': 'Compact Cars', 'Capacity': 110, 'Value': 1000}, {'VehicleID': 8, 'VehicleType': 'Luxury Sedans', 'Capacity': 40, 'Value': 3500}, {'VehicleID': 9, 'VehicleType': 'Vans', 'Capacity': 60, 'Value': 1600}, {'VehicleID': 10, 'VehicleType': 'Pickup Trucks', 'Capacity': 35, 'Value': 1700}]
vehicle_ids = [v['VehicleID'] for v in vehicle_types]
vehicle_names = [v['VehicleType'] for v in vehicle_types]
benefit_coefficients = {v['VehicleID']: v['Value'] for v in vehicle_types}
daily_inventory_limits = {v['VehicleID']: v['Capacity'] for v in vehicle_types}
total_inventory_capacity = sum(daily_inventory_limits.values())
m = gp.Model('Car_Inventory_Optimization')
x_vars = m.addVars(vehicle_ids, vtype=GRB.INTEGER, lb=0, name='')
for vid in vehicle_ids:
    m.addConstr(x_vars[vid] <= daily_inventory_limits[vid], name=f'cap_{vid}')
m.addConstr(gp.quicksum((x_vars[vid] for vid in vehicle_ids)) <= total_inventory_capacity, name='total_cap')
m.setObjective(gp.quicksum((benefit_coefficients[vid] * x_vars[vid] for vid in vehicle_ids)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for vid in vehicle_ids:
        print(f'x[{vid}]: {x_vars[vid].X}')
else:
    print(f'Solver status: {m.Status}')