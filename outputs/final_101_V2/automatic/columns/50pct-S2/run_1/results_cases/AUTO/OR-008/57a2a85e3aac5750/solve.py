LEGACY_OBSERVATION = 'products.csv\nwarranty_months,ProductName,Value\n48,Sedans,1200\n48,SUVs,1800\n60,Electric Vehicles,2500\n60,Hybrid Vehicles,2000\n12,Trucks,1500\n36,Sports Cars,3000\n36,Compact Cars,1000\n60,Luxury Sedans,3500\n48,Vans,1600\n60,Pickup Trucks,1700\n\ncapacity.csv\nshowroom_service_tier,annual_maintenance_visits,VehicleID,VehicleType,Capacity\nStandard,6,1,Sedans,100\nPriority,4,2,SUVs,80\nPriority,3,3,Electric Vehicles,120\nStandard,2,4,Hybrid Vehicles,90\nPremium,3,5,Trucks,50\nPriority,4,6,Sports Cars,30\nPriority,6,7,Compact Cars,110\nPriority,8,8,Luxury Sedans,40\nStandard,2,9,Vans,60\nStandard,2,10,Pickup Trucks,35'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'warranty_months': '48', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': 'products.csv', 'values': {'warranty_months': '48', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': 'products.csv', 'values': {'warranty_months': '60', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': 'products.csv', 'values': {'warranty_months': '60', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': 'products.csv', 'values': {'warranty_months': '12', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': 'products.csv', 'values': {'warranty_months': '36', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': 'products.csv', 'values': {'warranty_months': '36', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': 'products.csv', 'values': {'warranty_months': '60', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': 'products.csv', 'values': {'warranty_months': '48', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': 'products.csv', 'values': {'warranty_months': '60', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '6', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '4', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '3', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Premium', 'annual_maintenance_visits': '3', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '4', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '6', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Priority', 'annual_maintenance_visits': '8', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'showroom_service_tier': 'Standard', 'annual_maintenance_visits': '2', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}]
import gurobipy as gp
from gurobipy import GRB
products = [r['values'] for r in LEGACY_RECORDS if r['source'] == 'products.csv']
capacities = [r['values'] for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
vehicle_types = [p['ProductName'] for p in products]
benefit = {p['ProductName']: int(p['Value']) for p in products}
capacity = {c['VehicleType']: int(c['Capacity']) for c in capacities}
if set(vehicle_types) != set(capacity.keys()):
    raise ValueError('Mismatch between vehicle types in products and capacity records.')
if set(vehicle_types) != set(benefit.keys()):
    raise ValueError('Mismatch between vehicle types in products and benefit records.')
total_capacity = sum((capacity[vt] for vt in vehicle_types))
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vt] * x[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
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