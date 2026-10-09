CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of New Car Sales in Norway, a car dealership is planning its inventory-replenishment '
          'strategy. For each vehicle type (e.g., sedans, SUVs, electric vehicles, etc.), the dealership has a '
          '“products.csv” file that records the benefit coefficient for that type. Each vehicle type has a daily '
          'inventory limit, provided in “capacity.csv.” The objective is to decide how many units of each vehicle type '
          'to order each day so as to maximize total benefit while ensuring that the sum of all ordered units does not '
          'exceed the total inventory capacity. The decision variable x_i represents the number of vehicles of type i '
          'to be ordered per day.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['VehicleType', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
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
             'role': 'vehicle type daily inventory limits',
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
             'role': 'vehicle type benefit coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cap_records = CSVQA_DATA['tables'][0]['records']
    prod_records = CSVQA_DATA['tables'][1]['records']
    vehicle_types_cap = [rec['values']['VehicleType'] for rec in cap_records]
    vehicle_types_prod = [rec['values']['ProductName'] for rec in prod_records]
    I = [vt for vt in vehicle_types_cap if vt in vehicle_types_prod]
    u_i = {}
    for rec in cap_records:
        vt = rec['values']['VehicleType']
        if vt in I:
            u_i[vt] = int(rec['values']['Capacity'])
    b_i = {}
    for rec in prod_records:
        vt = rec['values']['ProductName']
        if vt in I:
            b_i[vt] = int(rec['values']['Value'])
    C = sum((u_i[vt] for vt in I))
    if set(I) != set(u_i.keys()) or set(I) != set(b_i.keys()):
        raise ValueError('Missing data for some vehicle types.')
    m = gp.Model('Car_Dealership_Inventory')
    x = m.addVars(I, lb=0, ub=[u_i[vt] for vt in I], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b_i[vt] * x[vt] for vt in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x[vt] for vt in I)) <= C, name='total_capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')