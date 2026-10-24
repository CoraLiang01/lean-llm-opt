CSVQA_DATA = {'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a “products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i to be ordered per day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['VehicleType', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'VehicleType': 'Sedans'}}, {'data_row_number': 1, 'values': {'Capacity': '80', 'VehicleType': 'SUVs'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'VehicleType': 'Electric Vehicles'}}, {'data_row_number': 3, 'values': {'Capacity': '90', 'VehicleType': 'Hybrid Vehicles'}}, {'data_row_number': 4, 'values': {'Capacity': '50', 'VehicleType': 'Trucks'}}, {'data_row_number': 5, 'values': {'Capacity': '30', 'VehicleType': 'Sports Cars'}}, {'data_row_number': 6, 'values': {'Capacity': '110', 'VehicleType': 'Compact Cars'}}, {'data_row_number': 7, 'values': {'Capacity': '40', 'VehicleType': 'Luxury Sedans'}}, {'data_row_number': 8, 'values': {'Capacity': '60', 'VehicleType': 'Vans'}}, {'data_row_number': 9, 'values': {'Capacity': '35', 'VehicleType': 'Pickup Trucks'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800'}}, {'data_row_number': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'data_row_number': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'data_row_number': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500'}}, {'data_row_number': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000'}}, {'data_row_number': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000'}}, {'data_row_number': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'data_row_number': 8, 'values': {'ProductName': 'Vans', 'Value': '1600'}}, {'data_row_number': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    val_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'), None)
    if cap_table is None or val_table is None:
        raise ValueError('Required tables not found in CSVQA_DATA')
    cap = {}
    for record in cap_table['records']:
        vehicle = record['values']['VehicleType']
        capacity = int(record['values']['Capacity'])
        cap[vehicle] = capacity
    val = {}
    for record in val_table['records']:
        product = record['values']['ProductName']
        value = int(record['values']['Value'])
        val[product] = value
    ids_cap = set(cap.keys())
    ids_val = set(val.keys())
    if ids_cap != ids_val:
        missing_in_cap = ids_val - ids_cap
        missing_in_val = ids_cap - ids_val
        msg_parts = []
        if missing_in_cap:
            msg_parts.append(f'missing in capacity table: {missing_in_cap}')
        if missing_in_val:
            msg_parts.append(f'missing in value table: {missing_in_val}')
        raise ValueError('Index mismatch: ' + '; '.join(msg_parts))
    ids = list(ids_cap)
    C_tot = sum((cap[i] for i in ids))
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(ids, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((val[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    for i in ids:
        m.addConstr(x[i] <= cap[i], name='')
    m.addConstr(gurobipy.quicksum((x[i] for i in ids)) <= C_tot, name='')
    m.optimize()
    return m
m = solve_problem()
