CSVQA_DATA = {'query': "In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment strategy.For each vehicle type, the dealership has a “products.csv” file that records the benefit coefficient for that type, in other word, the value for each car.There are some warehouse that can store these cars.Each warehosue has a capacity limit, provided in “capacity.csv.”The objective is to decide how many units of each vehicle type to store in the warehouse so as to maximize total value while ensuring that each warehouse won't exceed capacity limit.The decision variable x_i represents the number of vehicles of type i to be ordered per day. The decision variables must be integers.", 'relationships': [], 'tables': [{'columns': ['Warehouse ID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'Warehouse ID': 'Warehouse 1'}}, {'data_row_number': 1, 'values': {'Capacity': '80', 'Warehouse ID': 'Warehouse 2'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'Warehouse ID': 'Warehouse 3'}}, {'data_row_number': 3, 'values': {'Capacity': '90', 'Warehouse ID': 'Warehouse 4'}}, {'data_row_number': 4, 'values': {'Capacity': '50', 'Warehouse ID': 'Warehouse 5'}}, {'data_row_number': 5, 'values': {'Capacity': '30', 'Warehouse ID': 'Warehouse 6'}}, {'data_row_number': 6, 'values': {'Capacity': '110', 'Warehouse ID': 'Warehouse 7'}}, {'data_row_number': 7, 'values': {'Capacity': '40', 'Warehouse ID': 'Warehouse 8'}}, {'data_row_number': 8, 'values': {'Capacity': '60', 'Warehouse ID': 'Warehouse 9'}}, {'data_row_number': 9, 'values': {'Capacity': '35', 'Warehouse ID': 'Warehouse 10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}}, {'data_row_number': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}}, {'data_row_number': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}}, {'data_row_number': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}}, {'data_row_number': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}}, {'data_row_number': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}}, {'data_row_number': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}}, {'data_row_number': 8, 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}}, {'data_row_number': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    product_records = CSVQA_DATA['tables'][1]['records']
    warehouse_records = CSVQA_DATA['tables'][0]['records']
    I = [rec['data_row_number'] for rec in product_records]
    J = [rec['data_row_number'] for rec in warehouse_records]
    product_value = {}
    product_weight = {}
    for rec in product_records:
        key = rec['data_row_number']
        try:
            val = float(rec['values']['Value'])
            wgt = float(rec['values']['Weight'])
        except KeyError as e:
            raise ValueError(f'Missing parameter in product record {key}: {e}')
        product_value[key] = val
        product_weight[key] = wgt
    warehouse_capacity = {}
    for rec in warehouse_records:
        key = rec['data_row_number']
        try:
            cap = float(rec['values']['Capacity'])
        except KeyError as e:
            raise ValueError(f'Missing parameter in warehouse record {key}: {e}')
        warehouse_capacity[key] = cap
    m = gurobipy.Model('Warehouse_Inventory')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((product_value[i] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((product_weight[i] * x[i, j] for i in I)) <= warehouse_capacity[j] for j in J), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
