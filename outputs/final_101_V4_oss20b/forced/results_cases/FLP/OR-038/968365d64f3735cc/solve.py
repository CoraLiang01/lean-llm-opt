CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a '
          '“products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily '
          'inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type '
          'to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not '
          'exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i '
          'to be ordered per day.The decision variables must be integers.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['VehicleID', 'VehicleType', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '100', 'VehicleID': '1', 'VehicleType': 'Sedans'}},
                         {'source_row': 1, 'values': {'Capacity': '80', 'VehicleID': '2', 'VehicleType': 'SUVs'}},
                         {'source_row': 2,
                          'values': {'Capacity': '120', 'VehicleID': '3', 'VehicleType': 'Electric Vehicles'}},
                         {'source_row': 3,
                          'values': {'Capacity': '90', 'VehicleID': '4', 'VehicleType': 'Hybrid Vehicles'}},
                         {'source_row': 4, 'values': {'Capacity': '50', 'VehicleID': '5', 'VehicleType': 'Trucks'}},
                         {'source_row': 5,
                          'values': {'Capacity': '30', 'VehicleID': '6', 'VehicleType': 'Sports Cars'}},
                         {'source_row': 6,
                          'values': {'Capacity': '110', 'VehicleID': '7', 'VehicleType': 'Compact Cars'}},
                         {'source_row': 7,
                          'values': {'Capacity': '40', 'VehicleID': '8', 'VehicleType': 'Luxury Sedans'}},
                         {'source_row': 8, 'values': {'Capacity': '60', 'VehicleID': '9', 'VehicleType': 'Vans'}},
                         {'source_row': 9,
                          'values': {'Capacity': '35', 'VehicleID': '10', 'VehicleType': 'Pickup Trucks'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200'}},
                         {'source_row': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800'}},
                         {'source_row': 2, 'values': {'ProductName': 'Electric Vehicles', 'Value': '2500'}},
                         {'source_row': 3, 'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000'}},
                         {'source_row': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500'}},
                         {'source_row': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000'}},
                         {'source_row': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000'}},
                         {'source_row': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500'}},
                         {'source_row': 8, 'values': {'ProductName': 'Vans', 'Value': '1600'}},
                         {'source_row': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    benefit_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    cap = {}
    for rec in cap_table['records']:
        key = rec['values']['VehicleType']
        val = float(rec['values']['Capacity'])
        cap[key] = val
    benefit = {}
    for rec in benefit_table['records']:
        key = rec['values']['ProductName']
        val = float(rec['values']['Value'])
        benefit[key] = val
    V = set(cap.keys()).union(benefit.keys())
    missing_in_cap = V - set(cap.keys())
    missing_in_benefit = V - set(benefit.keys())
    if missing_in_cap:
        raise ValueError(f'Missing capacity for vehicle types: {missing_in_cap}')
    if missing_in_benefit:
        raise ValueError(f'Missing benefit for vehicle types: {missing_in_benefit}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(V, vtype=GRB.INTEGER, lb=0, name='')
    for v in V:
        m.addConstr(x[v] <= cap[v], name=f'cap_{v}')
    total_cap = sum((cap[v] for v in V))
    m.addConstr(gurobipy.quicksum((x[v] for v in V)) <= total_cap, name='total_cap')
    m.setObjective(gurobipy.quicksum((benefit[v] * x[v] for v in V)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    return m
m = solve_problem()