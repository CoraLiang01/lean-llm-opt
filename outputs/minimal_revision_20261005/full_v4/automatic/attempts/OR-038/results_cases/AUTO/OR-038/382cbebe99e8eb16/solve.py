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
    data = globals().get('CSVQA_DATA')
    if data is None:
        raise RuntimeError('CSVQA_DATA not found in global scope.')
    cap_table = None
    prod_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            cap_table = t
        elif t['table_id'] == 'file_1_view_0':
            prod_table = t
    if cap_table is None or prod_table is None:
        raise RuntimeError('Required tables not found in CSVQA_DATA.')
    cap_records = cap_table['records']
    prod_records = prod_table['records']
    vehicle_types = [r['values']['VehicleType'] for r in cap_records]
    product_names = [r['values']['ProductName'] for r in prod_records]
    I = [vt for vt in vehicle_types if vt in product_names]
    b_i = {}
    u_i = {}
    for vt in I:
        found_b = False
        for r in prod_records:
            if r['values']['ProductName'] == vt:
                try:
                    b_i[vt] = float(r['values']['Value'])
                except Exception:
                    raise ValueError(f'Invalid Value for {vt} in products.csv')
                found_b = True
                break
        if not found_b:
            raise ValueError(f'Benefit coefficient missing for {vt}')
        found_u = False
        for r in cap_records:
            if r['values']['VehicleType'] == vt:
                try:
                    u_i[vt] = int(r['values']['Capacity'])
                except Exception:
                    raise ValueError(f'Invalid Capacity for {vt} in capacity.csv')
                found_u = True
                break
        if not found_u:
            raise ValueError(f'Capacity missing for {vt}')
    C = sum((u_i[vt] for vt in I))
    m = gp.Model('Car_Inventory_Replenishment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, ub=[u_i[vt] for vt in I], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b_i[vt] * x[vt] for vt in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x[vt] for vt in I)) <= C, name='total_capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')