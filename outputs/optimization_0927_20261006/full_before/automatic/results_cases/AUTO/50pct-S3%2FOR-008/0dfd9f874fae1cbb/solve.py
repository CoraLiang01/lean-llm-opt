LEGACY_OBSERVATION = '{"previous_period_inventory_status": "Stockout", "previous_period_capacity": "110", "VehicleID": "1", "VehicleType": "Sedans", "Capacity": "100"}\n{"previous_period_inventory_status": "Overstock", "previous_period_capacity": "73", "VehicleID": "2", "VehicleType": "SUVs", "Capacity": "80"}\n{"previous_period_inventory_status": "Overstock", "previous_period_capacity": "141", "VehicleID": "3", "VehicleType": "Electric Vehicles", "Capacity": "120"}\n{"previous_period_inventory_status": "Overstock", "previous_period_capacity": "87", "VehicleID": "4", "VehicleType": "Hybrid Vehicles", "Capacity": "90"}\n{"previous_period_inventory_status": "Stockout", "previous_period_capacity": "60", "VehicleID": "5", "VehicleType": "Trucks", "Capacity": "50"}\n{"previous_period_inventory_status": "Balanced", "previous_period_capacity": "27", "VehicleID": "6", "VehicleType": "Sports Cars", "Capacity": "30"}\n{"previous_period_inventory_status": "Overstock", "previous_period_capacity": "116", "VehicleID": "7", "VehicleType": "Compact Cars", "Capacity": "110"}\n{"previous_period_inventory_status": "Overstock", "previous_period_capacity": "33", "VehicleID": "8", "VehicleType": "Luxury Sedans", "Capacity": "40"}\n{"previous_period_inventory_status": "Balanced", "previous_period_capacity": "61", "VehicleID": "9", "VehicleType": "Vans", "Capacity": "60"}\n{"previous_period_inventory_status": "Stockout", "previous_period_capacity": "32", "VehicleID": "10", "VehicleType": "Pickup Trucks", "Capacity": "35"}\n{"previous_period_unit_value": "1366", "ProductName": "Sedans", "Value": "1200"}\n{"previous_period_unit_value": "2006", "ProductName": "SUVs", "Value": "1800"}\n{"previous_period_unit_value": "2088", "ProductName": "Electric Vehicles", "Value": "2500"}\n{"previous_period_unit_value": "1875", "ProductName": "Hybrid Vehicles", "Value": "2000"}\n{"previous_period_unit_value": "1640", "ProductName": "Trucks", "Value": "1500"}\n{"previous_period_unit_value": "3292", "ProductName": "Sports Cars", "Value": "3000"}\n{"previous_period_unit_value": "854", "ProductName": "Compact Cars", "Value": "1000"}\n{"previous_period_unit_value": "3375", "ProductName": "Luxury Sedans", "Value": "3500"}\n{"previous_period_unit_value": "1783", "ProductName": "Vans", "Value": "1600"}\n{"previous_period_unit_value": "1874", "ProductName": "Pickup Trucks", "Value": "1700"}'
LEGACY_RECORDS = [{'source': '', 'values': {'previous_period_inventory_status': 'Stockout', 'previous_period_capacity': '110', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '73', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '141', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '87', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Stockout', 'previous_period_capacity': '60', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Balanced', 'previous_period_capacity': '27', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '116', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '33', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Balanced', 'previous_period_capacity': '61', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': '', 'values': {'previous_period_inventory_status': 'Stockout', 'previous_period_capacity': '32', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}, {'source': '', 'values': {'previous_period_unit_value': '1366', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': '', 'values': {'previous_period_unit_value': '2006', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': '', 'values': {'previous_period_unit_value': '2088', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': '', 'values': {'previous_period_unit_value': '1875', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': '', 'values': {'previous_period_unit_value': '1640', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': '', 'values': {'previous_period_unit_value': '3292', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': '', 'values': {'previous_period_unit_value': '854', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': '', 'values': {'previous_period_unit_value': '3375', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': '', 'values': {'previous_period_unit_value': '1783', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': '', 'values': {'previous_period_unit_value': '1874', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}]
import gurobipy as gp
from gurobipy import GRB
vehicle_info = []
vehicle_ids = []
vehicle_types = []
per_type_capacity = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'VehicleID' in v and 'VehicleType' in v and ('Capacity' in v):
        vid = str(v['VehicleID'])
        vehicle_ids.append(vid)
        vehicle_types.append(v['VehicleType'])
        per_type_capacity[vid] = int(v['Capacity'])
        vehicle_info.append({'VehicleID': vid, 'VehicleType': v['VehicleType']})
vehicletype_to_id = {t: vid for (vid, t) in zip(vehicle_ids, vehicle_types)}
value_coeff = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'ProductName' in v and 'Value' in v:
        vt = v['ProductName']
        val = int(v['Value'])
        if vt in vehicletype_to_id:
            value_coeff[vehicletype_to_id[vt]] = val
for vid in vehicle_ids:
    if vid not in value_coeff:
        raise ValueError(f'Missing value coefficient for VehicleID {vid}')
    if vid not in per_type_capacity:
        raise ValueError(f'Missing capacity for VehicleID {vid}')
total_capacity = sum((per_type_capacity[vid] for vid in vehicle_ids))
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_ids, lb=0, ub=[per_type_capacity[vid] for vid in vehicle_ids], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value_coeff[vid] * x[vid] for vid in vehicle_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[vid] for vid in vehicle_ids)) <= total_capacity, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')