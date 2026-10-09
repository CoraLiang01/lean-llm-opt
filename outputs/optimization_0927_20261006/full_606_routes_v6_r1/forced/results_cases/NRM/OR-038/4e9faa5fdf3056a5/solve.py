CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a '
          '“products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily '
          'inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type '
          'to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not '
          'exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i '
          'to be ordered per day.The decision variables must be integers.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['VehicleID', 'VehicleType', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'inventory/capacity limits',
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
             'role': 'benefit coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
capacity_table = None
products_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        capacity_table = t
    elif t['table_id'] == 'file_1_view_0':
        products_table = t
if capacity_table is None or products_table is None:
    raise ValueError('Required tables not found in CSVQA_DATA.')
vehicle_types = []
capacity_dict = {}
for rec in capacity_table['records']:
    vt = rec['values']['VehicleType']
    vehicle_types.append(vt)
    try:
        capacity_dict[vt] = int(rec['values']['Capacity'])
    except Exception:
        raise ValueError(f'Invalid capacity for vehicle type {vt}')
benefit_dict = {}
for rec in products_table['records']:
    vt = rec['values']['ProductName']
    try:
        benefit_dict[vt] = float(rec['values']['Value'])
    except Exception:
        raise ValueError(f'Invalid benefit coefficient for vehicle type {vt}')
for vt in vehicle_types:
    if vt not in benefit_dict:
        raise ValueError(f'Vehicle type {vt} missing benefit coefficient in products.csv')
    if vt not in capacity_dict:
        raise ValueError(f'Vehicle type {vt} missing capacity in capacity.csv')
C = sum((capacity_dict[vt] for vt in vehicle_types))
m = gp.Model('NewCarSales_NRM')
x_vars = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit_dict[vt] * x_vars[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
m.addConstrs((x_vars[vt] <= capacity_dict[vt] for vt in vehicle_types), name='')
m.addConstr(gp.quicksum((x_vars[vt] for vt in vehicle_types)) <= C, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')