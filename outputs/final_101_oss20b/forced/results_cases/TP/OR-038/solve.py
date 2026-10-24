CSVQA_DATA = {'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a “products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i to be ordered per day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['VehicleID', 'VehicleType', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'VehicleID': '1', 'VehicleType': 'Sedans'}}, {'data_row_number': 1, 'values': {'Capacity': '80', 'VehicleID': '2', 'VehicleType': 'SUVs'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles'}}, {'data_row_number': 3, 'values': {'Capacity': '90', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles'}}, {'data_row_number': 4, 'values': {'Capacity': '50', 'VehicleID': '5', 'VehicleType': 'Trucks'}}, {'data_row_number': 5, 'values': {'Capacity': '30', 'VehicleID': '6', 'VehicleType': 'Sports Cars'}}, {'data_row_number': 6, 'values': {'Capacity': '110', 'VehicleID': '7', 'VehicleType': 'Compact Cars'}}, {'data_row_number': 7, 'values': {'Capacity': '40', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans'}}, {'data_row_number': 8, 'values': {'Capacity': '60', 'VehicleID': '9', 'VehicleType': 'Vans'}}, {'data_row_number': 9, 'values': {'Capacity': '35', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}}, {'data_row_number': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}}, {'data_row_number': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}}, {'data_row_number': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}}, {'data_row_number': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}}, {'data_row_number': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}}, {'data_row_number': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}}, {'data_row_number': 8, 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}}, {'data_row_number': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    cap = {}
    vehicle_type_by_id = {}
    for rec in cap_table['records']:
        vid = rec['values']['VehicleID']
        cap_val = int(rec['values']['Capacity'])
        cap[vid] = cap_val
        vehicle_type_by_id[vid] = rec['values']['VehicleType']
    val_by_name = {}
    for rec in prod_table['records']:
        pname = rec['values']['ProductName']
        val_by_name[pname] = int(rec['values']['Value'])
    val = {}
    for vid, vtype in vehicle_type_by_id.items():
        if vtype not in val_by_name:
            raise ValueError(f'Missing value for vehicle type {vtype}')
        val[vid] = val_by_name[vtype]
    for vid in cap:
        if vid not in val:
            raise ValueError(f'Missing value for vehicle ID {vid}')
    I = [rec['values']['VehicleID'] for rec in cap_table['records']]
    C = sum((cap[vid] for vid in I))
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((val[vid] * x[vid] for vid in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((x[vid] for vid in I)) <= C, name='total_capacity')
    m.addConstrs((x[vid] <= cap[vid] for vid in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
