LEGACY_OBSERVATION = 'products.csv\n\nProductName,Value,previous_period_unit_value,two_periods_ago_unit_value\nSedans,1200,1366,1388\nSUVs,1800,2006,2070\nElectric Vehicles,2500,2088,2528\nHybrid Vehicles,2000,1875,1748\nTrucks,1500,1640,1744\nSports Cars,3000,3292,3584\nCompact Cars,1000,854,1090\nLuxury Sedans,3500,3375,3809\nVans,1600,1783,1481\nPickup Trucks,1700,1874,1432\n\ncapacity.csv\n\nVehicleID,VehicleType,Capacity,previous_period_capacity,capacity_two_periods_ago,previous_period_inventory_status\n1,Sedans,100,110,116,Stockout\n2,SUVs,80,73,66,Overstock\n3,Electric Vehicles,120,141,123,Overstock\n4,Hybrid Vehicles,90,87,80,Overstock\n5,Trucks,50,60,41,Stockout\n6,Sports Cars,30,27,32,Balanced\n7,Compact Cars,110,116,109,Overstock\n8,Luxury Sedans,40,33,43,Overstock\n9,Vans,60,61,69,Balanced\n10,Pickup Trucks,35,32,36,Stockout'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'ProductName': 'Sedans', 'Value': '1200', 'previous_period_unit_value': '1366', 'two_periods_ago_unit_value': '1388'}}, {'source': 'products.csv', 'values': {'ProductName': 'SUVs', 'Value': '1800', 'previous_period_unit_value': '2006', 'two_periods_ago_unit_value': '2070'}}, {'source': 'products.csv', 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'previous_period_unit_value': '2088', 'two_periods_ago_unit_value': '2528'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'previous_period_unit_value': '1875', 'two_periods_ago_unit_value': '1748'}}, {'source': 'products.csv', 'values': {'ProductName': 'Trucks', 'Value': '1500', 'previous_period_unit_value': '1640', 'two_periods_ago_unit_value': '1744'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'previous_period_unit_value': '3292', 'two_periods_ago_unit_value': '3584'}}, {'source': 'products.csv', 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'previous_period_unit_value': '854', 'two_periods_ago_unit_value': '1090'}}, {'source': 'products.csv', 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'previous_period_unit_value': '3375', 'two_periods_ago_unit_value': '3809'}}, {'source': 'products.csv', 'values': {'ProductName': 'Vans', 'Value': '1600', 'previous_period_unit_value': '1783', 'two_periods_ago_unit_value': '1481'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'previous_period_unit_value': '1874', 'two_periods_ago_unit_value': '1432'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100', 'previous_period_capacity': '110', 'capacity_two_periods_ago': '116', 'previous_period_inventory_status': 'Stockout'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80', 'previous_period_capacity': '73', 'capacity_two_periods_ago': '66', 'previous_period_inventory_status': 'Overstock'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120', 'previous_period_capacity': '141', 'capacity_two_periods_ago': '123', 'previous_period_inventory_status': 'Overstock'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90', 'previous_period_capacity': '87', 'capacity_two_periods_ago': '80', 'previous_period_inventory_status': 'Overstock'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50', 'previous_period_capacity': '60', 'capacity_two_periods_ago': '41', 'previous_period_inventory_status': 'Stockout'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30', 'previous_period_capacity': '27', 'capacity_two_periods_ago': '32', 'previous_period_inventory_status': 'Balanced'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110', 'previous_period_capacity': '116', 'capacity_two_periods_ago': '109', 'previous_period_inventory_status': 'Overstock'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40', 'previous_period_capacity': '33', 'capacity_two_periods_ago': '43', 'previous_period_inventory_status': 'Overstock'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60', 'previous_period_capacity': '61', 'capacity_two_periods_ago': '69', 'previous_period_inventory_status': 'Balanced'}}, {'source': 'capacity.csv', 'values': {'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35', 'previous_period_capacity': '32', 'capacity_two_periods_ago': '36', 'previous_period_inventory_status': 'Stockout'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
values = {}
for rec in records:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        try:
            values[pname] = int(rec['values']['Value'])
        except Exception:
            raise ValueError(f'Missing or invalid Value for product {pname}')
capacities = {}
capacity_list = []
for rec in records:
    if rec['source'] == 'capacity.csv':
        pname = rec['values']['VehicleType']
        try:
            cap = int(rec['values']['Capacity'])
        except Exception:
            raise ValueError(f'Missing or invalid Capacity for vehicle {pname}')
        capacities[pname] = cap
        capacity_list.append(cap)
if len(products) != len(capacities):
    raise ValueError('Mismatch between number of products and capacities')
for pname in products:
    if pname not in capacities:
        raise ValueError(f'Missing capacity for product {pname}')
    if pname not in values:
        raise ValueError(f'Missing value for product {pname}')
total_capacity = sum(capacity_list)
m = gp.Model('Car_Dealership_Inventory')
x = m.addVars(products, lb=0, ub={p: capacities[p] for p in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[p] for p in products)) <= total_capacity, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')