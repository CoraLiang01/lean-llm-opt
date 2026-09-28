LEGACY_OBSERVATION = '{"values": {"VehicleID": "1", "VehicleType": "Sedans", "Capacity": "100"}}\n{"values": {"VehicleID": "2", "VehicleType": "SUVs", "Capacity": "80"}}\n{"values": {"VehicleID": "3", "VehicleType": "Electric Vehicles", "Capacity": "120"}}\n{"values": {"VehicleID": "4", "VehicleType": "Hybrid Vehicles", "Capacity": "90"}}\n{"values": {"VehicleID": "5", "VehicleType": "Trucks", "Capacity": "50"}}\n{"values": {"VehicleID": "6", "VehicleType": "Sports Cars", "Capacity": "30"}}\n{"values": {"VehicleID": "7", "VehicleType": "Compact Cars", "Capacity": "110"}}\n{"values": {"VehicleID": "8", "VehicleType": "Luxury Sedans", "Capacity": "40"}}\n{"values": {"VehicleID": "9", "VehicleType": "Vans", "Capacity": "60"}}\n{"values": {"VehicleID": "10", "VehicleType": "Pickup Trucks", "Capacity": "35"}}\n{"values": {"ProductName": "Sedans", "Value": "1200", "Weight": "20"}}\n{"values": {"ProductName": "SUVs", "Value": "1800", "Weight": "15"}}\n{"values": {"ProductName": "Electric Vehicles", "Value": "2500", "Weight": "25"}}\n{"values": {"ProductName": "Hybrid Vehicles", "Value": "2000", "Weight": "18"}}\n{"values": {"ProductName": "Trucks", "Value": "1500", "Weight": "10"}}\n{"values": {"ProductName": "Sports Cars", "Value": "3000", "Weight": "5"}}\n{"values": {"ProductName": "Compact Cars", "Value": "1000", "Weight": "22"}}\n{"values": {"ProductName": "Luxury Sedans", "Value": "3500", "Weight": "8"}}\n{"values": {"ProductName": "Vans", "Value": "1600", "Weight": "12"}}\n{"values": {"ProductName": "Pickup Trucks", "Value": "1700", "Weight": "7"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': '', 'values': {'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': '', 'values': {'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': '', 'values': {'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': '', 'values': {'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': '', 'values': {'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': '', 'values': {'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': '', 'values': {'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': '', 'values': {'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': '', 'values': {'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}, {'source': '', 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}}, {'source': '', 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}}, {'source': '', 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}}, {'source': '', 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}}, {'source': '', 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}}, {'source': '', 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}}, {'source': '', 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}}, {'source': '', 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}}, {'source': '', 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}}, {'source': '', 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}]
import gurobipy as gp
from gurobipy import GRB
vehicle_caps = {}
vehicle_types = {}
vehicle_ids = []
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'VehicleID' in v and 'VehicleType' in v and ('Capacity' in v):
        vid = str(v['VehicleID'])
        vehicle_ids.append(vid)
        vehicle_types[vid] = v['VehicleType']
        vehicle_caps[vid] = int(v['Capacity'])
benefit = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'ProductName' in v and 'Value' in v:
        pname = v['ProductName']
        val = int(v['Value'])
        for vid, vtype in vehicle_types.items():
            if vtype == pname:
                benefit[vid] = val
for vid in vehicle_ids:
    if vid not in benefit:
        raise ValueError(f'Missing benefit for VehicleID {vid}')
    if vid not in vehicle_caps:
        raise ValueError(f'Missing capacity for VehicleID {vid}')
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_ids, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vid] * x[vid] for vid in vehicle_ids)), GRB.MAXIMIZE)
m.addConstrs((x[vid] <= vehicle_caps[vid] for vid in vehicle_ids), name='')
total_capacity = sum((vehicle_caps[vid] for vid in vehicle_ids))
m.addConstr(gp.quicksum((x[vid] for vid in vehicle_ids)) <= total_capacity, name='total_cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')