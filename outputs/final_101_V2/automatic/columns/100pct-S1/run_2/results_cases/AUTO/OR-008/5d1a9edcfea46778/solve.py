LEGACY_OBSERVATION = 'products.csv\narchived_attachment_count,archive_revision_number,ProductName,Value\n2,7,Sedans,1200\n2,6,SUVs,1800\n1,7,Electric Vehicles,2500\n3,2,Hybrid Vehicles,2000\n3,2,Trucks,1500\n6,5,Sports Cars,3000\n1,4,Compact Cars,1000\n4,6,Luxury Sedans,3500\n1,5,Vans,1600\n2,5,Pickup Trucks,1700\n\ncapacity.csv\nrecord_keeper_group,archive_revision_number,archived_attachment_count,VehicleID,VehicleType,Capacity\nTeam B,8,6,1,Sedans,100\nTeam A,7,1,2,SUVs,80\nTeam A,8,1,3,Electric Vehicles,120\nTeam A,8,6,4,Hybrid Vehicles,90\nTeam B,6,3,5,Trucks,50\nTeam C,6,3,6,Sports Cars,30\nTeam A,8,4,7,Compact Cars,110\nTeam A,2,6,8,Luxury Sedans,40\nTeam C,3,1,9,Vans,60\nTeam B,6,1,10,Pickup Trucks,35'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'archived_attachment_count': '2', 'archive_revision_number': '7', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '2', 'archive_revision_number': '6', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '1', 'archive_revision_number': '7', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '3', 'archive_revision_number': '2', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '3', 'archive_revision_number': '2', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '6', 'archive_revision_number': '5', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '1', 'archive_revision_number': '4', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '4', 'archive_revision_number': '6', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '1', 'archive_revision_number': '5', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '2', 'archive_revision_number': '5', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '8', 'archived_attachment_count': '6', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '7', 'archived_attachment_count': '1', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'archived_attachment_count': '1', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'archived_attachment_count': '6', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'archived_attachment_count': '3', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '6', 'archived_attachment_count': '3', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'archived_attachment_count': '4', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '2', 'archived_attachment_count': '6', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '3', 'archived_attachment_count': '1', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'archived_attachment_count': '1', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}]
import gurobipy as gp
from gurobipy import GRB
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
product_value = {}
for rec in product_records:
    name = rec['values']['ProductName']
    val = rec['values']['Value']
    if name in product_value:
        raise ValueError(f'Duplicate product value for {name}')
    product_value[name] = int(val)
vehicle_capacity = {}
for rec in capacity_records:
    name = rec['values']['VehicleType']
    cap = rec['values']['Capacity']
    if name in vehicle_capacity:
        raise ValueError(f'Duplicate capacity for {name}')
    vehicle_capacity[name] = int(cap)
vehicle_types = list(product_value.keys())
if set(vehicle_types) != set(vehicle_capacity.keys()):
    raise ValueError('Mismatch between product and capacity vehicle types')
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((product_value[v] * x[v] for v in vehicle_types)), GRB.MAXIMIZE)
for v in vehicle_types:
    m.addConstr(x[v] <= vehicle_capacity[v], name=f'cap_{v}')
total_capacity = sum((vehicle_capacity[v] for v in vehicle_types))
m.addConstr(gp.quicksum((x[v] for v in vehicle_types)) <= total_capacity, name='total_cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')