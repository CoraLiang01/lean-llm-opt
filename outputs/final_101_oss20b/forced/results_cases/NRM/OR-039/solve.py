CSVQA_DATA = {'query': "In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment strategy.For each vehicle type, the dealership has a “products.csv” file that records the benefit coefficient for that type, in other word, the value for each car.There are some warehouse that can store these cars.Each warehosue has a capacity limit, provided in “capacity.csv.”The objective is to decide how many units of each vehicle type to store in the warehouse so as to maximize total value while ensuring that each warehouse won't exceed capacity limit.The decision variable x_i represents the number of vehicles of type i to be ordered per day. The decision variables must be integers.", 'relationships': [], 'tables': [{'columns': ['Warehouse ID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'Warehouse ID': 'Warehouse 1'}}, {'data_row_number': 1, 'values': {'Capacity': '80', 'Warehouse ID': 'Warehouse 2'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'Warehouse ID': 'Warehouse 3'}}, {'data_row_number': 3, 'values': {'Capacity': '90', 'Warehouse ID': 'Warehouse 4'}}, {'data_row_number': 4, 'values': {'Capacity': '50', 'Warehouse ID': 'Warehouse 5'}}, {'data_row_number': 5, 'values': {'Capacity': '30', 'Warehouse ID': 'Warehouse 6'}}, {'data_row_number': 6, 'values': {'Capacity': '110', 'Warehouse ID': 'Warehouse 7'}}, {'data_row_number': 7, 'values': {'Capacity': '40', 'Warehouse ID': 'Warehouse 8'}}, {'data_row_number': 8, 'values': {'Capacity': '60', 'Warehouse ID': 'Warehouse 9'}}, {'data_row_number': 9, 'values': {'Capacity': '35', 'Warehouse ID': 'Warehouse 10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800'}}, {'data_row_number': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'data_row_number': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'data_row_number': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500'}}, {'data_row_number': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000'}}, {'data_row_number': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000'}}, {'data_row_number': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'data_row_number': 8, 'values': {'ProductName': 'Vans', 'Value': '1600'}}, {'data_row_number': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    product_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    W = []
    capacity = {}
    for rec in capacity_table['records']:
        w_id = rec['values']['Warehouse ID']
        cap_val = int(rec['values']['Capacity'])
        if w_id in capacity:
            raise ValueError(f'Duplicate Warehouse ID {w_id} in capacity data')
        W.append(w_id)
        capacity[w_id] = cap_val
    P = []
    value = {}
    for rec in product_table['records']:
        p_id = rec['values']['ProductName']
        val_val = int(rec['values']['Value'])
        if p_id in value:
            raise ValueError(f'Duplicate ProductName {p_id} in product data')
        P.append(p_id)
        value[p_id] = val_val
    for w in W:
        if w not in capacity:
            raise ValueError(f'Missing capacity for warehouse {w}')
    for p in P:
        if p not in value:
            raise ValueError(f'Missing value for product {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(w, p) for w in W for p in P], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((value[p] * x[w, p] for w in W for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((x[w, p] for p in P)) <= capacity[w] for w in W), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
