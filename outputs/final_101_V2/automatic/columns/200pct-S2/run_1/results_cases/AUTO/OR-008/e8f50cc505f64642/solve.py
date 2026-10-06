LEGACY_OBSERVATION = 'products.csv\nProductName,registration_document_pages,vehicle_catalog_views_last_month,warranty_months,average_test_drive_minutes,Value\nSedans,6,680,48,15,1200\nSUVs,10,920,48,30,1800\nElectric Vehicles,10,920,60,15,2500\nHybrid Vehicles,10,680,60,30,2000\nTrucks,8,120,12,25,1500\nSports Cars,6,250,36,40,3000\nCompact Cars,12,410,36,25,1000\nLuxury Sedans,12,120,60,15,3500\nVans,8,120,48,30,1600\nPickup Trucks,12,250,60,20,1700\n\ncapacity.csv\nVehicleID,VehicleType,Capacity\n1,Sedans,100\n2,SUVs,80\n3,Electric Vehicles,120\n4,Hybrid Vehicles,90\n5,Trucks,50\n6,Sports Cars,30\n7,Compact Cars,110\n8,Luxury Sedans,40\n9,Vans,60\n10,Pickup Trucks,35'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'ProductName': 'Sedans', 'registration_document_pages': '6', 'vehicle_catalog_views_last_month': '680', 'warranty_months': '48', 'average_test_drive_minutes': '15', 'Value': '1200'}}, {'source': 'products.csv', 'values': {'ProductName': 'SUVs', 'registration_document_pages': '10', 'vehicle_catalog_views_last_month': '920', 'warranty_months': '48', 'average_test_drive_minutes': '30', 'Value': '1800'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric Vehicles', 'registration_document_pages': '10', 'vehicle_catalog_views_last_month': '920', 'warranty_months': '60', 'average_test_drive_minutes': '15', 'Value': '2500'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid Vehicles', 'registration_document_pages': '10', 'vehicle_catalog_views_last_month': '680', 'warranty_months': '60', 'average_test_drive_minutes': '30', 'Value': '2000'}}, {'source': 'products.csv', 'values': {'ProductName': 'Trucks', 'registration_document_pages': '8', 'vehicle_catalog_views_last_month': '120', 'warranty_months': '12', 'average_test_drive_minutes': '25', 'Value': '1500'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports Cars', 'registration_document_pages': '6', 'vehicle_catalog_views_last_month': '250', 'warranty_months': '36', 'average_test_drive_minutes': '40', 'Value': '3000'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact Cars', 'registration_document_pages': '12', 'vehicle_catalog_views_last_month': '410', 'warranty_months': '36', 'average_test_drive_minutes': '25', 'Value': '1000'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury Sedans', 'registration_document_pages': '12', 'vehicle_catalog_views_last_month': '120', 'warranty_months': '60', 'average_test_drive_minutes': '15', 'Value': '3500'}}, {'source': 'products.csv', 'values': {'ProductName': 'Vans', 'registration_document_pages': '8', 'vehicle_catalog_views_last_month': '120', 'warranty_months': '48', 'average_test_drive_minutes': '30', 'Value': '1600'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pickup Trucks', 'registration_document_pages': '12', 'vehicle_catalog_views_last_month': '250', 'warranty_months': '60', 'average_test_drive_minutes': '20', 'Value': '1700'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
vehicle_types = []
benefit = {}
capacity = {}
for rec in records:
    if rec['source'] == 'products.csv':
        name = rec['values']['ProductName']
        vehicle_types.append(name)
        benefit[name] = int(rec['values']['Value'])
    elif rec['source'] == 'capacity.csv':
        name = rec['values']['VehicleType']
        capacity[name] = int(rec['values']['Capacity'])
for vt in vehicle_types:
    if vt not in benefit:
        raise ValueError(f'Missing benefit for vehicle type {vt}')
    if vt not in capacity:
        raise ValueError(f'Missing capacity for vehicle type {vt}')
total_capacity = sum((capacity[vt] for vt in vehicle_types))
m = gp.Model('Car_Dealership_Inventory')
x = m.addVars(vehicle_types, lb=0, ub=[capacity[vt] for vt in vehicle_types], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vt] * x[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[vt] for vt in vehicle_types)) <= total_capacity, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for vt in vehicle_types:
        print(f'x[{vt}]: {x[vt].X}')
else:
    print(f'Solver status: {m.Status}')