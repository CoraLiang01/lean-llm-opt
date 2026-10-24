CSVQA_DATA = {'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a “products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i to be ordered per day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['VehicleType', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'VehicleType': 'Sedans'}}, {'data_row_number': 1, 'values': {'Capacity': '80', 'VehicleType': 'SUVs'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'VehicleType': 'Electric Vehicles'}}, {'data_row_number': 3, 'values': {'Capacity': '90', 'VehicleType': 'Hybrid Vehicles'}}, {'data_row_number': 4, 'values': {'Capacity': '50', 'VehicleType': 'Trucks'}}, {'data_row_number': 5, 'values': {'Capacity': '30', 'VehicleType': 'Sports Cars'}}, {'data_row_number': 6, 'values': {'Capacity': '110', 'VehicleType': 'Compact Cars'}}, {'data_row_number': 7, 'values': {'Capacity': '40', 'VehicleType': 'Luxury Sedans'}}, {'data_row_number': 8, 'values': {'Capacity': '60', 'VehicleType': 'Vans'}}, {'data_row_number': 9, 'values': {'Capacity': '35', 'VehicleType': 'Pickup Trucks'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800'}}, {'data_row_number': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'data_row_number': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'data_row_number': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500'}}, {'data_row_number': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000'}}, {'data_row_number': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000'}}, {'data_row_number': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'data_row_number': 8, 'values': {'ProductName': 'Vans', 'Value': '1600'}}, {'data_row_number': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB
tables = CSVQA_DATA['tables']
capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
benefit_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
capacity_dict = {}
for record in capacity_table['records']:
    vehicle = record['values']['VehicleType']
    cap_str = record['values']['Capacity']
    try:
        cap = int(cap_str)
    except ValueError:
        raise ValueError(f"Non‑numeric capacity value for vehicle '{vehicle}': {cap_str}")
    capacity_dict[vehicle] = cap
benefit_dict = {}
for record in benefit_table['records']:
    vehicle = record['values']['ProductName']
    val_str = record['values']['Value']
    try:
        val = int(val_str)
    except ValueError:
        raise ValueError(f"Non‑numeric benefit value for vehicle '{vehicle}': {val_str}")
    benefit_dict[vehicle] = val
vehicle_types = set(capacity_dict.keys()).union(benefit_dict.keys())
for vt in vehicle_types:
    if vt not in capacity_dict:
        raise ValueError(f"Missing capacity for vehicle type '{vt}'.")
    if vt not in benefit_dict:
        raise ValueError(f"Missing benefit for vehicle type '{vt}'.")
total_capacity = sum((capacity_dict[vt] for vt in vehicle_types))

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((benefit_dict[vt] * x[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
    for vt in vehicle_types:
        m.addConstr(x[vt] <= capacity_dict[vt], name=f'cap_{vt}')
    m.addConstr(gurobipy.quicksum((x[vt] for vt in vehicle_types)) <= total_capacity, name='total_cap')
    m.optimize()
    return m
m = solve_problem()
