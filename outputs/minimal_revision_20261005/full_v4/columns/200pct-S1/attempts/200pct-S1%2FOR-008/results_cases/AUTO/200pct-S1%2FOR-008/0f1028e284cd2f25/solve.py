CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a '
          '‚Äúproducts.csv‚Äù file that records the benefit coefficient for that type. Each vehicle type has a daily '
          'inventory limit, provided in ‚Äúcapacity.csv.‚Äù The objective is to decide how many units of each vehicle '
          'type to order each day so as to maximize total benefit while ensuring that the sum of all ordered units '
          'does not exceed the total inventory capacity. The decision variable x_i represents the number of vehicles '
          'of type i to be ordered per day.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['VehicleType', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'Capacity': '100', 'VehicleType': 'Sedans'}},
                         {'source_row': 1, 'values': {'Capacity': '80', 'VehicleType': 'SUVs'}},
                         {'source_row': 2, 'values': {'Capacity': '120', 'VehicleType': 'Electric Vehicles'}},
                         {'source_row': 3, 'values': {'Capacity': '90', 'VehicleType': 'Hybrid Vehicles'}},
                         {'source_row': 4, 'values': {'Capacity': '50', 'VehicleType': 'Trucks'}},
                         {'source_row': 5, 'values': {'Capacity': '30', 'VehicleType': 'Sports Cars'}},
                         {'source_row': 6, 'values': {'Capacity': '110', 'VehicleType': 'Compact Cars'}},
                         {'source_row': 7, 'values': {'Capacity': '40', 'VehicleType': 'Luxury Sedans'}},
                         {'source_row': 8, 'values': {'Capacity': '60', 'VehicleType': 'Vans'}},
                         {'source_row': 9, 'values': {'Capacity': '35', 'VehicleType': 'Pickup Trucks'}}],
             'returned_rows': 10,
             'role': 'vehicle capacity per type',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'benefit coefficient per vehicle type',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = globals()['CSVQA_DATA']
    table_capacity = None
    table_benefit = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            table_capacity = t
        elif t['table_id'] == 'file_1_view_0':
            table_benefit = t
    if table_capacity is None or table_benefit is None:
        raise RuntimeError('Required tables not found in CSVQA_DATA.')
    vehicle_types_capacity = [rec['values']['VehicleType'] for rec in table_capacity['records']]
    vehicle_types_benefit = [rec['values']['ProductName'] for rec in table_benefit['records']]
    I = sorted(set(vehicle_types_capacity) & set(vehicle_types_benefit))
    b_i = {}
    u_i = {}
    benefit_map = {rec['values']['ProductName']: rec['values']['Value'] for rec in table_benefit['records']}
    capacity_map = {rec['values']['VehicleType']: rec['values']['Capacity'] for rec in table_capacity['records']}
    for i in I:
        if i not in benefit_map or i not in capacity_map:
            raise ValueError(f'Missing data for vehicle type {i}')
        try:
            b_i[i] = float(benefit_map[i])
            u_i[i] = int(capacity_map[i])
        except Exception as e:
            raise ValueError(f'Invalid data for vehicle type {i}: {e}')
    m = gp.Model('Car_Inventory_Replenishment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, ub=[u_i[i] for i in I], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    total_capacity = sum((u_i[i] for i in I))
    m.addConstr(gp.quicksum((x[i] for i in I)) <= total_capacity, name='total_inventory_capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()