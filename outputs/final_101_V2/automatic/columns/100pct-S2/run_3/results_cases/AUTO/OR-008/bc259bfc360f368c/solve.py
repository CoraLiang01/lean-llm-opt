LEGACY_OBSERVATION = 'products.csv\nvehicle_catalog_views_last_month,warranty_months,ProductName,Value\n680,48,Sedans,1200\n920,48,SUVs,1800\n920,60,Electric Vehicles,2500\n680,60,Hybrid Vehicles,2000\n120,12,Trucks,1500\n250,36,Sports Cars,3000\n410,36,Compact Cars,1000\n120,60,Luxury Sedans,3500\n120,48,Vans,1600\n250,60,Pickup Trucks,1700\n\ncapacity.csv\nshowroom_service_tier,annual_maintenance_visits,vehicle_bay_cleaning_minutes_last_month,VehicleID,VehicleType,Capacity\nStandard,6,180,1,Sedans,100\nPriority,4,240,2,SUVs,80\nPriority,3,300,3,Electric Vehicles,120\nStandard,2,360,4,Hybrid Vehicles,90\nPremium,3,120,5,Trucks,50\nPriority,4,180,6,Sports Cars,30\nPriority,6,360,7,Compact Cars,110\nPriority,8,240,8,Luxury Sedans,40\nStandard,2,240,9,Vans,60\nStandard,2,240,10,Pickup Trucks,35'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '680', 'warranty_months': '48', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '920', 'warranty_months': '48', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '920', 'warranty_months': '60', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '680', 'warranty_months': '60', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '120', 'warranty_months': '12', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '250', 'warranty_months': '36', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '410', 'warranty_months': '36', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '120', 'warranty_months': '60', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '120', 'warranty_months': '48', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': 'products.csv', 'values': {'vehicle_catalog_views_last_month': '250', 'warranty_months': '60', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '6', 'vehicle_bay_cleaning_minutes_last_month': '180', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '4', 'vehicle_bay_cleaning_minutes_last_month': '240', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '3', 'vehicle_bay_cleaning_minutes_last_month': '300', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'vehicle_bay_cleaning_minutes_last_month': '360', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Premium', 'annual_maintenance_visits': '3', 'vehicle_bay_cleaning_minutes_last_month': '120', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '4', 'vehicle_bay_cleaning_minutes_last_month': '180', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '6', 'vehicle_bay_cleaning_minutes_last_month': '360', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '8', 'vehicle_bay_cleaning_minutes_last_month': '240', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'vehicle_bay_cleaning_minutes_last_month': '240', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'vehicle_bay_cleaning_minutes_last_month': '240', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}]
import gurobipy as gp
from gurobipy import GRB
vehicle_types = []
value = {}
capacity = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        vt = rec['values']['ProductName']
        vehicle_types.append(vt)
        value[vt] = int(rec['values']['Value'])
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        vt = rec['values']['VehicleType']
        capacity[vt] = int(rec['values']['Capacity'])
for vt in vehicle_types:
    if vt not in value or vt not in capacity:
        raise ValueError(f'Missing value or capacity for vehicle type: {vt}')
total_capacity = sum((capacity[vt] for vt in vehicle_types))
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[vt] * x[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
m.addConstrs((x[vt] <= capacity[vt] for vt in vehicle_types), name='')
m.addConstr(gp.quicksum((x[vt] for vt in vehicle_types)) <= total_capacity, name='total_cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')