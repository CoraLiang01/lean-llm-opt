LEGACY_OBSERVATION = 'products.csv\narchive_revision_number,ProductName,Value\n7,Sedans,1200\n6,SUVs,1800\n7,Electric Vehicles,2500\n2,Hybrid Vehicles,2000\n2,Trucks,1500\n5,Sports Cars,3000\n4,Compact Cars,1000\n6,Luxury Sedans,3500\n5,Vans,1600\n5,Pickup Trucks,1700\n\ncapacity.csv\nrecord_keeper_group,archive_revision_number,VehicleID,VehicleType,Capacity\nTeam B,8,1,Sedans,100\nTeam A,7,2,SUVs,80\nTeam A,8,3,Electric Vehicles,120\nTeam A,8,4,Hybrid Vehicles,90\nTeam B,6,5,Trucks,50\nTeam C,6,6,Sports Cars,30\nTeam A,8,7,Compact Cars,110\nTeam A,2,8,Luxury Sedans,40\nTeam C,3,9,Vans,60\nTeam B,6,10,Pickup Trucks,35'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'archive_revision_number': '7', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '6', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '7', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '2', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '2', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '5', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '4', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '6', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '5', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '5', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '8', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '7', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '6', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '2', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '3', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
benefit = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        benefit[pname] = int(rec['values']['Value'])
capacities = {}
vehicle_types = []
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        vtype = rec['values']['VehicleType']
        vehicle_types.append(vtype)
        capacities[vtype] = int(rec['values']['Capacity'])
if set(products) != set(vehicle_types):
    raise ValueError('Mismatch between products and vehicle types in records.')
vehicle_types = products.copy()
total_capacity = sum((capacities[vtype] for vtype in vehicle_types))
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_types, lb=0, ub=[capacities[vtype] for vtype in vehicle_types], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vtype] * x[vtype] for vtype in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[vtype] for vtype in vehicle_types)) <= total_capacity, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')