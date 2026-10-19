CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a '
          '“products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily '
          'inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type '
          'to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not '
          'exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i '
          'to be ordered per day.The decision variables must be integers.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'aliases': ['file_0_view_0', 'capacity.csv'],
             'columns': ['VehicleID', 'VehicleType', 'Capacity'],
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
            {'aliases': ['file_1_view_0', 'products.csv'],
             'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Sedans', 'Value': '1200', 'Weight': '20'}},
                         {'source_row': 1, 'values': {'ProductName': 'SUVs', 'Value': '1800', 'Weight': '15'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Electric Vehicles', 'Value': '2500', 'Weight': '25'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Hybrid Vehicles', 'Value': '2000', 'Weight': '18'}},
                         {'source_row': 4, 'values': {'ProductName': 'Trucks', 'Value': '1500', 'Weight': '10'}},
                         {'source_row': 5, 'values': {'ProductName': 'Sports Cars', 'Value': '3000', 'Weight': '5'}},
                         {'source_row': 6, 'values': {'ProductName': 'Compact Cars', 'Value': '1000', 'Weight': '22'}},
                         {'source_row': 7, 'values': {'ProductName': 'Luxury Sedans', 'Value': '3500', 'Weight': '8'}},
                         {'source_row': 8, 'values': {'ProductName': 'Vans', 'Value': '1600', 'Weight': '12'}},
                         {'source_row': 9, 'values': {'ProductName': 'Pickup Trucks', 'Value': '1700', 'Weight': '7'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
capacity_table = None
products_table = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0' or 'capacity.csv' in tbl.get('aliases', []):
        capacity_table = tbl
    if tbl.get('table_id') == 'file_1_view_0' or 'products.csv' in tbl.get('aliases', []):
        products_table = tbl
type_to_value = {}
for rec in products_table['records']:
    vt = rec['values']['ProductName']
    val = float(rec['values']['Value'])
    type_to_value[vt] = val
vehicle_ids = []
capacities = []
benefits = []
for rec in capacity_table['records']:
    vid = rec['values']['VehicleID']
    vt = rec['values']['VehicleType']
    cap = int(rec['values']['Capacity'])
    vehicle_ids.append(vid)
    capacities.append(cap)
    benefits.append(type_to_value.get(vt, 0.0))
m = gp.Model('Car_Inventory_Replenishment')
m.setParam(GRB.Param.MIPGap, 1e-06)
x = {}
for vid, cap in zip(vehicle_ids, capacities):
    var = m.addVar(vtype=GRB.INTEGER, lb=0, ub=cap)
    x[vid] = var
m.setObjective(gp.quicksum((benefits[i] * x[vehicle_ids[i]] for i in range(len(vehicle_ids)))), GRB.MAXIMIZE)
for vid, cap in zip(vehicle_ids, capacities):
    m.addConstr(x[vid] <= cap)
total_capacity = sum(capacities)
m.addConstr(gp.quicksum((x[vid] for vid in vehicle_ids)) <= total_capacity)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal}')
    for vid in vehicle_ids:
        var = x[vid]
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.Status}')