CSVQA_DATA = {'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a “products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i to be ordered per day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['VehicleType', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'VehicleType': 'Sedans'}}, {'data_row_number': 1, 'values': {'Capacity': '80', 'VehicleType': 'SUVs'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'VehicleType': 'Electric Vehicles'}}, {'data_row_number': 3, 'values': {'Capacity': '90', 'VehicleType': 'Hybrid Vehicles'}}, {'data_row_number': 4, 'values': {'Capacity': '50', 'VehicleType': 'Trucks'}}, {'data_row_number': 5, 'values': {'Capacity': '30', 'VehicleType': 'Sports Cars'}}, {'data_row_number': 6, 'values': {'Capacity': '110', 'VehicleType': 'Compact Cars'}}, {'data_row_number': 7, 'values': {'Capacity': '40', 'VehicleType': 'Luxury Sedans'}}, {'data_row_number': 8, 'values': {'Capacity': '60', 'VehicleType': 'Vans'}}, {'data_row_number': 9, 'values': {'Capacity': '35', 'VehicleType': 'Pickup Trucks'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800'}}, {'data_row_number': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'data_row_number': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'data_row_number': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500'}}, {'data_row_number': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000'}}, {'data_row_number': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000'}}, {'data_row_number': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'data_row_number': 8, 'values': {'ProductName': 'Vans', 'Value': '1600'}}, {'data_row_number': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    product_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    capacity_dict = {}
    for rec in capacity_table['records']:
        key = rec['values']['VehicleType']
        val = int(rec['values']['Capacity'])
        capacity_dict[key] = val
    value_dict = {}
    for rec in product_table['records']:
        key = rec['values']['ProductName']
        val = int(rec['values']['Value'])
        value_dict[key] = val
    for k in capacity_dict:
        if k not in value_dict:
            raise ValueError(f"Missing Value for VehicleType '{k}'")
    for k in value_dict:
        if k not in capacity_dict:
            raise ValueError(f"Missing Capacity for ProductName '{k}'")
    I = list(capacity_dict.keys())
    total_capacity = sum((capacity_dict[i] for i in I))
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    for i in I:
        model.addConstr(x[i] <= capacity_dict[i], name='')
    model.addConstr(gurobipy.quicksum((x[i] for i in I)) <= total_capacity, name='')
    model.setObjective(gurobipy.quicksum((value_dict[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
