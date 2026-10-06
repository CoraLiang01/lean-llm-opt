LEGACY_OBSERVATION = 'products.csv\nProductName,vehicle_catalog_views_last_month,warranty_months,Value\nSedans,680,48,1200\nSUVs,920,48,1800\nElectric Vehicles,920,60,2500\nHybrid Vehicles,680,60,2000\nTrucks,120,12,1500\nSports Cars,250,36,3000\nCompact Cars,410,36,1000\nLuxury Sedans,120,60,3500\nVans,120,48,1600\nPickup Trucks,250,60,1700\n\ncapacity.csv\nVehicleID,VehicleType,Capacity\n1,Sedans,100\n2,SUVs,80\n3,Electric Vehicles,120\n4,Hybrid Vehicles,90\n5,Trucks,50\n6,Sports Cars,30\n7,Compact Cars,110\n8,Luxury Sedans,40\n9,Vans,60\n10,Pickup Trucks,35'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'ProductName': 'Sedans', 'vehicle_catalog_views_last_month': '680', 'warranty_months': '48', 'Value': '1200'}}, {'source': 'products.csv', 'values': {'ProductName': 'SUVs', 'vehicle_catalog_views_last_month': '920', 'warranty_months': '48', 'Value': '1800'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric Vehicles', 'vehicle_catalog_views_last_month': '920', 'warranty_months': '60', 'Value': '2500'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid Vehicles', 'vehicle_catalog_views_last_month': '680', 'warranty_months': '60', 'Value': '2000'}}, {'source': 'products.csv', 'values': {'ProductName': 'Trucks', 'vehicle_catalog_views_last_month': '120', 'warranty_months': '12', 'Value': '1500'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports Cars', 'vehicle_catalog_views_last_month': '250', 'warranty_months': '36', 'Value': '3000'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact Cars', 'vehicle_catalog_views_last_month': '410', 'warranty_months': '36', 'Value': '1000'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury Sedans', 'vehicle_catalog_views_last_month': '120', 'warranty_months': '60', 'Value': '3500'}}, {'source': 'products.csv', 'values': {'ProductName': 'Vans', 'vehicle_catalog_views_last_month': '120', 'warranty_months': '48', 'Value': '1600'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pickup Trucks', 'vehicle_catalog_views_last_month': '250', 'warranty_months': '60', 'Value': '1700'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}]
import gurobipy as gp
from gurobipy import GRB
vehicle_types = []
value = {}
capacity = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        name = rec['values']['ProductName']
        vehicle_types.append(name)
        value[name] = int(rec['values']['Value'])
    elif rec['source'] == 'capacity.csv':
        name = rec['values']['VehicleType']
        capacity[name] = int(rec['values']['Capacity'])
if set(vehicle_types) != set(capacity.keys()):
    raise ValueError('Mismatch between vehicle types in products.csv and capacity.csv')
if set(vehicle_types) != set(value.keys()):
    raise ValueError('Mismatch between vehicle types in products.csv and value dictionary')
total_capacity = sum((capacity[v] for v in vehicle_types))
m = gp.Model('Car_Dealership_Inventory')
x = m.addVars(vehicle_types, lb=0, ub=[capacity[v] for v in vehicle_types], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[v] * x[v] for v in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[v] for v in vehicle_types)) <= total_capacity, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in vehicle_types:
        print(f'x[{v}]: {x[v].X}')
else:
    print(f'Solver status: {m.Status}')