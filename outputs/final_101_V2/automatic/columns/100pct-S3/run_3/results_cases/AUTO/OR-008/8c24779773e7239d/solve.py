LEGACY_OBSERVATION = 'products.csv\n\ntwo_periods_ago_unit_value,previous_period_unit_value,ProductName,Value\n1388,1366,Sedans,1200\n2070,2006,SUVs,1800\n2528,2088,Electric Vehicles,2500\n1748,1875,Hybrid Vehicles,2000\n1744,1640,Trucks,1500\n3584,3292,Sports Cars,3000\n1090,854,Compact Cars,1000\n3809,3375,Luxury Sedans,3500\n1481,1783,Vans,1600\n1432,1874,Pickup Trucks,1700\n\ncapacity.csv\n\nprevious_period_inventory_status,previous_period_capacity,capacity_two_periods_ago,VehicleID,VehicleType,Capacity\nStockout,110,116,1,Sedans,100\nOverstock,73,66,2,SUVs,80\nOverstock,141,123,3,Electric Vehicles,120\nOverstock,87,80,4,Hybrid Vehicles,90\nStockout,60,41,5,Trucks,50\nBalanced,27,32,6,Sports Cars,30\nOverstock,116,109,7,Compact Cars,110\nOverstock,33,43,8,Luxury Sedans,40\nBalanced,61,69,9,Vans,60\nStockout,32,36,10,Pickup Trucks,35'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '1388', 'previous_period_unit_value': '1366', 'ProductName': 'Sedans', 'Value': '1200'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '2070', 'previous_period_unit_value': '2006', 'ProductName': 'SUVs', 'Value': '1800'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '2528', 'previous_period_unit_value': '2088', 'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '1748', 'previous_period_unit_value': '1875', 'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '1744', 'previous_period_unit_value': '1640', 'ProductName': 'Trucks', 'Value': '1500'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '3584', 'previous_period_unit_value': '3292', 'ProductName': 'Sports Cars', 'Value': '3000'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '1090', 'previous_period_unit_value': '854', 'ProductName': 'Compact Cars', 'Value': '1000'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '3809', 'previous_period_unit_value': '3375', 'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '1481', 'previous_period_unit_value': '1783', 'ProductName': 'Vans', 'Value': '1600'}}, {'source': 'products.csv', 'values': {'two_periods_ago_unit_value': '1432', 'previous_period_unit_value': '1874', 'ProductName': 'Pickup Trucks', 'Value': '1700'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Stockout', 'previous_period_capacity': '110', 'capacity_two_periods_ago': '116', 'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '73', 'capacity_two_periods_ago': '66', 'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '141', 'capacity_two_periods_ago': '123', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '87', 'capacity_two_periods_ago': '80', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Stockout', 'previous_period_capacity': '60', 'capacity_two_periods_ago': '41', 'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Balanced', 'previous_period_capacity': '27', 'capacity_two_periods_ago': '32', 'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '116', 'capacity_two_periods_ago': '109', 'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Overstock', 'previous_period_capacity': '33', 'capacity_two_periods_ago': '43', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Balanced', 'previous_period_capacity': '61', 'capacity_two_periods_ago': '69', 'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': 'capacity.csv', 'values': {'previous_period_inventory_status': 'Stockout', 'previous_period_capacity': '32', 'capacity_two_periods_ago': '36', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}]
import gurobipy as gp
from gurobipy import GRB
vehicle_types = []
benefit = {}
capacity = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        vt = rec['values']['ProductName']
        vehicle_types.append(vt)
        benefit[vt] = int(rec['values']['Value'])
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        vt = rec['values']['VehicleType']
        capacity[vt] = int(rec['values']['Capacity'])
if set(vehicle_types) != set(benefit.keys()) or set(vehicle_types) != set(capacity.keys()):
    raise ValueError('Mismatch in vehicle types between products.csv and capacity.csv')
total_capacity = sum((capacity[vt] for vt in vehicle_types))
m = gp.Model('Car_Inventory_Optimization')
x = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vt] * x[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
m.addConstrs((x[vt] <= capacity[vt] for vt in vehicle_types), name='')
m.addConstr(gp.quicksum((x[vt] for vt in vehicle_types)) <= total_capacity, name='total_capacity')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for vt in vehicle_types:
        print(f'x[{vt}]: {x[vt].X}')
else:
    print(f'Solver status: {m.Status}')