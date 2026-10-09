LEGACY_OBSERVATION = '{"values": {"showroom_service_tier": "Standard", "annual_maintenance_visits": "6", "VehicleID": "1", "VehicleType": "Sedans", "Capacity": "100"}}\n{"values": {"showroom_service_tier": "Priority", "annual_maintenance_visits": "4", "VehicleID": "2", "VehicleType": "SUVs", "Capacity": "80"}}\n{"values": {"showroom_service_tier": "Priority", "annual_maintenance_visits": "3", "VehicleID": "3", "VehicleType": "Electric Vehicles", "Capacity": "120"}}\n{"values": {"showroom_service_tier": "Standard", "annual_maintenance_visits": "2", "VehicleID": "4", "VehicleType": "Hybrid Vehicles", "Capacity": "90"}}\n{"values": {"showroom_service_tier": "Premium", "annual_maintenance_visits": "3", "VehicleID": "5", "VehicleType": "Trucks", "Capacity": "50"}}\n{"values": {"showroom_service_tier": "Priority", "annual_maintenance_visits": "4", "VehicleID": "6", "VehicleType": "Sports Cars", "Capacity": "30"}}\n{"values": {"showroom_service_tier": "Priority", "annual_maintenance_visits": "6", "VehicleID": "7", "VehicleType": "Compact Cars", "Capacity": "110"}}\n{"values": {"showroom_service_tier": "Priority", "annual_maintenance_visits": "8", "VehicleID": "8", "VehicleType": "Luxury Sedans", "Capacity": "40"}}\n{"values": {"showroom_service_tier": "Standard", "annual_maintenance_visits": "2", "VehicleID": "9", "VehicleType": "Vans", "Capacity": "60"}}\n{"values": {"showroom_service_tier": "Standard", "annual_maintenance_visits": "2", "VehicleID": "10", "VehicleType": "Pickup Trucks", "Capacity": "35"}}\n{"values": {"warranty_months": "48", "ProductName": "Sedans", "Value": "1200"}}\n{"values": {"warranty_months": "48", "ProductName": "SUVs", "Value": "1800"}}\n{"values": {"warranty_months": "60", "ProductName": "Electric Vehicles", "Value": "2500"}}\n{"values": {"warranty_months": "60", "ProductName": "Hybrid Vehicles", "Value": "2000"}}\n{"values": {"warranty_months": "12", "ProductName": "Trucks", "Value": "1500"}}\n{"values": {"warranty_months": "36", "ProductName": "Sports Cars", "Value": "3000"}}\n{"values": {"warranty_months": "36", "ProductName": "Compact Cars", "Value": "1000"}}\n{"values": {"warranty_months": "60", "ProductName": "Luxury Sedans", "Value": "3500"}}\n{"values": {"warranty_months": "48", "ProductName": "Vans", "Value": "1600"}}\n{"values": {"warranty_months": "60", "ProductName": "Pickup Trucks", "Value": "1700"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '6', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': '', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '4', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': '', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '3', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': '', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': '', 'values': {'showroom_service_tier': 'Premium', 'annual_maintenance_visits': '3', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': '', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '4', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': '', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '6', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': '', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '8', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': '', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': '', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}, {'source': '', 'values': {'warranty_months': '48', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': '', 'values': {'warranty_months': '48', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': '', 'values': {'warranty_months': '60', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': '', 'values': {'warranty_months': '60', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': '', 'values': {'warranty_months': '12', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': '', 'values': {'warranty_months': '36', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': '', 'values': {'warranty_months': '36', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': '', 'values': {'warranty_months': '60', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': '', 'values': {'warranty_months': '48', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': '', 'values': {'warranty_months': '60', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}]
import gurobipy as gp
from gurobipy import GRB
vehicle_info = []
benefit_by_type = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'VehicleID' in v and 'VehicleType' in v and ('Capacity' in v):
        vehicle_info.append({'VehicleID': v['VehicleID'], 'VehicleType': v['VehicleType'], 'Capacity': int(v['Capacity'])})
    if 'ProductName' in v and 'Value' in v:
        benefit_by_type[v['ProductName']] = int(v['Value'])
vehicle_ids = []
vehicle_types = {}
capacity = {}
benefit = {}
for v in vehicle_info:
    vid = v['VehicleID']
    vtype = v['VehicleType']
    vehicle_ids.append(vid)
    vehicle_types[vid] = vtype
    capacity[vid] = v['Capacity']
    if vtype not in benefit_by_type:
        raise ValueError(f'Missing benefit coefficient for vehicle type {vtype}')
    benefit[vid] = benefit_by_type[vtype]
for vid in vehicle_ids:
    vtype = vehicle_types[vid]
    if vtype not in benefit_by_type:
        raise ValueError(f'Missing benefit coefficient for vehicle type {vtype}')
C = sum((capacity[vid] for vid in vehicle_ids))
m = gp.Model('Car_Dealership_Inventory')
x = m.addVars(vehicle_ids, lb=0, ub=[capacity[vid] for vid in vehicle_ids], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vid] * x[vid] for vid in vehicle_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[vid] for vid in vehicle_ids)) <= C, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for vid in vehicle_ids:
        print(f'x[{vid}]: {x[vid].X}')
else:
    print(f'Solver status: {m.Status}')