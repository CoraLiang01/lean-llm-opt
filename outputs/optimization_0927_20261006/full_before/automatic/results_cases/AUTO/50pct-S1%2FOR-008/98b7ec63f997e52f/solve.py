LEGACY_OBSERVATION = '{"values": {"record_keeper_group": "Team B", "archive_revision_number": "8", "VehicleID": "1", "VehicleType": "Sedans", "Capacity": "100"}}\n{"values": {"record_keeper_group": "Team A", "archive_revision_number": "7", "VehicleID": "2", "VehicleType": "SUVs", "Capacity": "80"}}\n{"values": {"record_keeper_group": "Team A", "archive_revision_number": "8", "VehicleID": "3", "VehicleType": "Electric Vehicles", "Capacity": "120"}}\n{"values": {"record_keeper_group": "Team A", "archive_revision_number": "8", "VehicleID": "4", "VehicleType": "Hybrid Vehicles", "Capacity": "90"}}\n{"values": {"record_keeper_group": "Team B", "archive_revision_number": "6", "VehicleID": "5", "VehicleType": "Trucks", "Capacity": "50"}}\n{"values": {"record_keeper_group": "Team C", "archive_revision_number": "6", "VehicleID": "6", "VehicleType": "Sports Cars", "Capacity": "30"}}\n{"values": {"record_keeper_group": "Team A", "archive_revision_number": "8", "VehicleID": "7", "VehicleType": "Compact Cars", "Capacity": "110"}}\n{"values": {"record_keeper_group": "Team A", "archive_revision_number": "2", "VehicleID": "8", "VehicleType": "Luxury Sedans", "Capacity": "40"}}\n{"values": {"record_keeper_group": "Team C", "archive_revision_number": "3", "VehicleID": "9", "VehicleType": "Vans", "Capacity": "60"}}\n{"values": {"record_keeper_group": "Team B", "archive_revision_number": "6", "VehicleID": "10", "VehicleType": "Pickup Trucks", "Capacity": "35"}}\n{"values": {"archive_revision_number": "7", "ProductName": "Sedans", "Value": "1200"}}\n{"values": {"archive_revision_number": "6", "ProductName": "SUVs", "Value": "1800"}}\n{"values": {"archive_revision_number": "7", "ProductName": "Electric Vehicles", "Value": "2500"}}\n{"values": {"archive_revision_number": "2", "ProductName": "Hybrid Vehicles", "Value": "2000"}}\n{"values": {"archive_revision_number": "2", "ProductName": "Trucks", "Value": "1500"}}\n{"values": {"archive_revision_number": "5", "ProductName": "Sports Cars", "Value": "3000"}}\n{"values": {"archive_revision_number": "4", "ProductName": "Compact Cars", "Value": "1000"}}\n{"values": {"archive_revision_number": "6", "ProductName": "Luxury Sedans", "Value": "3500"}}\n{"values": {"archive_revision_number": "5", "ProductName": "Vans", "Value": "1600"}}\n{"values": {"archive_revision_number": "5", "ProductName": "Pickup Trucks", "Value": "1700"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '8', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '7', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': '', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '6', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '8', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '2', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '3', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': '', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}, {'source': '', 'values': {'archive_revision_number': '7', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': '', 'values': {'archive_revision_number': '6', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': '', 'values': {'archive_revision_number': '7', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': '', 'values': {'archive_revision_number': '2', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': '', 'values': {'archive_revision_number': '2', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': '', 'values': {'archive_revision_number': '5', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': '', 'values': {'archive_revision_number': '4', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': '', 'values': {'archive_revision_number': '6', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': '', 'values': {'archive_revision_number': '5', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': '', 'values': {'archive_revision_number': '5', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}]
import gurobipy as gp
from gurobipy import GRB
vehicle_records = [r for r in LEGACY_RECORDS if 'VehicleID' in r['values']]
benefit_records = [r for r in LEGACY_RECORDS if 'ProductName' in r['values'] and 'Value' in r['values']]
vehicle_data = {}
for rec in vehicle_records:
    vid = int(rec['values']['VehicleID'])
    vtype = rec['values']['VehicleType']
    cap = int(rec['values']['Capacity'])
    vehicle_data[vid] = {'VehicleType': vtype, 'Capacity': cap}
benefit_data = {}
for rec in benefit_records:
    vtype = rec['values']['ProductName']
    val = int(rec['values']['Value'])
    benefit_data[vtype] = val
vehicles = []
for vid in sorted(vehicle_data):
    vtype = vehicle_data[vid]['VehicleType']
    cap = vehicle_data[vid]['Capacity']
    if vtype not in benefit_data:
        raise ValueError(f'Missing benefit coefficient for vehicle type {vtype}')
    val = benefit_data[vtype]
    vehicles.append({'VehicleID': vid, 'VehicleType': vtype, 'Capacity': cap, 'Value': val})
vehicle_ids = [v['VehicleID'] for v in vehicles]
capacity = {v['VehicleID']: v['Capacity'] for v in vehicles}
benefit = {v['VehicleID']: v['Value'] for v in vehicles}
C_total = sum((capacity[vid] for vid in vehicle_ids))
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_ids, lb=0, ub=[capacity[vid] for vid in vehicle_ids], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vid] * x[vid] for vid in vehicle_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[vid] for vid in vehicle_ids)) <= C_total, name='total_inventory')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')