SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '1', 'VehicleType': 'Sedans', 'Capacity': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '2', 'VehicleType': 'SUVs', 'Capacity': '80'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '3', 'VehicleType': 'Electric Vehicles', 'Capacity': '120'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '5', 'VehicleType': 'Trucks', 'Capacity': '50'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '6', 'VehicleType': 'Sports Cars', 'Capacity': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '7', 'VehicleType': 'Compact Cars', 'Capacity': '110'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '8', 'VehicleType': 'Luxury Sedans', 'Capacity': '40'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '9', 'VehicleType': 'Vans', 'Capacity': '60'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'values': {'VehicleID': '10', 'VehicleType': 'Pickup Trucks', 'Capacity': '35'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}]
import gurobipy as gp
from gurobipy import GRB
capacity_records = [r for r in SOURCE_RECORDS if r['source'].endswith('capacity.csv')]
product_records = [r for r in SOURCE_RECORDS if r['source'].endswith('products.csv')]
vehicle_info = []
for rec in capacity_records:
    vehicle_info.append({'VehicleID': int(rec['values']['VehicleID']), 'VehicleType': rec['values']['VehicleType'], 'Capacity': int(rec['values']['Capacity'])})
vehicle_info.sort(key=lambda x: x['VehicleID'])
benefit_map = {}
for rec in product_records:
    benefit_map[rec['values']['ProductName']] = int(rec['values']['Value'])
vehicle_ids = [v['VehicleID'] for v in vehicle_info]
vehicle_types = [v['VehicleType'] for v in vehicle_info]
capacity = {v['VehicleID']: v['Capacity'] for v in vehicle_info}
benefit = {}
for v in vehicle_info:
    vt = v['VehicleType']
    vid = v['VehicleID']
    if vt not in benefit_map:
        raise ValueError(f'Missing benefit coefficient for vehicle type {vt}')
    benefit[vid] = benefit_map[vt]
total_capacity = sum((capacity[vid] for vid in vehicle_ids))
m = gp.Model('NewCarSalesInNorway1')
x_vars = m.addVars(vehicle_ids, lb=0, ub=[capacity[vid] for vid in vehicle_ids], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[vid] * x_vars[vid] for vid in vehicle_ids)), GRB.MAXIMIZE)
m.addConstrs((x_vars[vid] <= capacity[vid] for vid in vehicle_ids), name='')
m.addConstr(gp.quicksum((x_vars[vid] for vid in vehicle_ids)) <= total_capacity, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for vid in vehicle_ids:
        print(f'x[{vid}]: {x_vars[vid].X}')
else:
    print(f'Solver status: {m.Status}')