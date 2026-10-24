CSVQA_DATA = {'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a “products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i to be ordered per day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['VehicleID', 'VehicleType', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'VehicleID': '1', 'VehicleType': 'Sedans'}}, {'data_row_number': 1, 'values': {'Capacity': '80', 'VehicleID': '2', 'VehicleType': 'SUVs'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles'}}, {'data_row_number': 3, 'values': {'Capacity': '90', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles'}}, {'data_row_number': 4, 'values': {'Capacity': '50', 'VehicleID': '5', 'VehicleType': 'Trucks'}}, {'data_row_number': 5, 'values': {'Capacity': '30', 'VehicleID': '6', 'VehicleType': 'Sports Cars'}}, {'data_row_number': 6, 'values': {'Capacity': '110', 'VehicleID': '7', 'VehicleType': 'Compact Cars'}}, {'data_row_number': 7, 'values': {'Capacity': '40', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans'}}, {'data_row_number': 8, 'values': {'Capacity': '60', 'VehicleID': '9', 'VehicleType': 'Vans'}}, {'data_row_number': 9, 'values': {'Capacity': '35', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200'}}, {'data_row_number': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800'}}, {'data_row_number': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500'}}, {'data_row_number': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000'}}, {'data_row_number': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500'}}, {'data_row_number': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000'}}, {'data_row_number': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000'}}, {'data_row_number': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500'}}, {'data_row_number': 8, 'values': {'ProductName': 'Vans', 'Value': '1600'}}, {'data_row_number': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    vehicle_info = {}
    for rec in cap_table['records']:
        vals = rec['values']
        vid = vals['VehicleID']
        vtype = vals['VehicleType']
        cap = int(vals['Capacity'])
        if vid in vehicle_info:
            raise ValueError(f'Duplicate VehicleID {vid} in capacity table')
        vehicle_info[vid] = {'VehicleType': vtype, 'Capacity': cap}
    C_tot = sum((info['Capacity'] for info in vehicle_info.values()))
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    product_value = {}
    for rec in prod_table['records']:
        vals = rec['values']
        pname = vals['ProductName']
        val = int(vals['Value'])
        if pname in product_value:
            raise ValueError(f'Duplicate ProductName {pname} in products table')
        product_value[pname] = val
    I = list(vehicle_info.keys())
    c_i = {}
    v_i = {}
    for vid in I:
        vtype = vehicle_info[vid]['VehicleType']
        if vtype not in product_value:
            raise ValueError(f'Missing value for VehicleType {vtype} (VehicleID {vid})')
        c_i[vid] = vehicle_info[vid]['Capacity']
        v_i[vid] = product_value[vtype]
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((v_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((x[i] for i in I)) <= C_tot, name='')
    m.addConstrs((x[i] <= c_i[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
