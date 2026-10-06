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
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    cap_table = None
    prod_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            cap_table = t
        elif t['table_id'] == 'file_1_view_0':
            prod_table = t
    if cap_table is None or prod_table is None:
        raise RuntimeError('Missing required tables.')
    prod_value = {}
    for rec in prod_table['records']:
        pname = rec['values']['ProductName']
        val = rec['values']['Value']
        try:
            prod_value[pname] = float(val)
        except Exception:
            raise RuntimeError(f'Invalid Value for product {pname}: {val}')
    I = []
    vehicle_type = {}
    u = {}
    for rec in cap_table['records']:
        vid = rec['values']['VehicleID']
        vtype = rec['values']['VehicleType']
        cap = rec['values']['Capacity']
        I.append(vid)
        vehicle_type[vid] = vtype
        try:
            u[vid] = int(cap)
        except Exception:
            raise RuntimeError(f'Invalid Capacity for VehicleID {vid}: {cap}')
    b = {}
    for vid in I:
        vtype = vehicle_type[vid]
        if vtype not in prod_value:
            raise RuntimeError(f'Missing Value for VehicleType/ProductName {vtype}')
        b[vid] = prod_value[vtype]
    C = sum((u[vid] for vid in I))
    m = gp.Model('Car_Inventory_Replenishment')
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b[vid] * x[vid] for vid in I)), GRB.MAXIMIZE)
    m.addConstrs((x[vid] <= u[vid] for vid in I), name='')
    m.addConstr(gp.quicksum((x[vid] for vid in I)) <= C, name='total_capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()