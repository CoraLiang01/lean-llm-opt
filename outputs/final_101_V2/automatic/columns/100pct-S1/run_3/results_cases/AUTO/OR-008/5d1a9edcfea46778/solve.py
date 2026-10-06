LEGACY_OBSERVATION = 'products.csv\narchived_attachment_count,archive_revision_number,ProductName,Value\n2,7,Sedans,1200\n2,6,SUVs,1800\n1,7,Electric Vehicles,2500\n3,2,Hybrid Vehicles,2000\n3,2,Trucks,1500\n6,5,Sports Cars,3000\n1,4,Compact Cars,1000\n4,6,Luxury Sedans,3500\n1,5,Vans,1600\n2,5,Pickup Trucks,1700\n\ncapacity.csv\nrecord_keeper_group,archive_revision_number,archived_attachment_count,VehicleID,VehicleType,Capacity\nTeam B,8,6,1,Sedans,100\nTeam A,7,1,2,SUVs,80\nTeam A,8,1,3,Electric Vehicles,120\nTeam A,8,6,4,Hybrid Vehicles,90\nTeam B,6,3,5,Trucks,50\nTeam C,6,3,6,Sports Cars,30\nTeam A,8,4,7,Compact Cars,110\nTeam A,2,6,8,Luxury Sedans,40\nTeam C,3,1,9,Vans,60\nTeam B,6,1,10,Pickup Trucks,35'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'archived_attachment_count': '2', 'archive_revision_number': '7', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '2', 'archive_revision_number': '6', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '1', 'archive_revision_number': '7', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '3', 'archive_revision_number': '2', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '3', 'archive_revision_number': '2', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '6', 'archive_revision_number': '5', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '1', 'archive_revision_number': '4', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '4', 'archive_revision_number': '6', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '1', 'archive_revision_number': '5', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': 'products.csv', 'values': {'archived_attachment_count': '2', 'archive_revision_number': '5', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '8', 'archived_attachment_count': '6', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '7', 'archived_attachment_count': '1', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'archived_attachment_count': '1', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'archived_attachment_count': '6', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'archived_attachment_count': '3', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '6', 'archived_attachment_count': '3', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'archived_attachment_count': '4', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '2', 'archived_attachment_count': '6', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '3', 'archived_attachment_count': '1', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'archived_attachment_count': '1', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = [r for r in records if r['source'] == 'products.csv']
capacities = [r for r in records if r['source'] == 'capacity.csv']
if len(products) != len(capacities):
    raise ValueError('Mismatch in number of products and capacities.')
n = len(products)
vehicle_types = []
values = {}
capacity = {}
for i in range(n):
    pname = products[i]['values']['ProductName']
    vtype = capacities[i]['values']['VehicleType']
    if pname != vtype:
        raise ValueError(f"ProductName '{pname}' does not match VehicleType '{vtype}' at index {i}.")
    vehicle_types.append(pname)
    try:
        values[pname] = int(products[i]['values']['Value'])
        capacity[pname] = int(capacities[i]['values']['Capacity'])
    except Exception as e:
        raise ValueError(f"Error parsing Value or Capacity for '{pname}': {e}")
total_capacity = sum((capacity[p] for p in vehicle_types))
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_types, lb=0, ub=[capacity[p] for p in vehicle_types], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[p] for p in vehicle_types)) <= total_capacity, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')