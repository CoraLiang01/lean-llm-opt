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
             'role': 'vehicle type capacity',
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
             'role': 'vehicle type benefit coefficient',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = {'capacity': [{'VehicleType': 'Sedans', 'Capacity': '100'}, {'VehicleType': 'SUVs', 'Capacity': '80'}, {'VehicleType': 'Electric Vehicles', 'Capacity': '120'}, {'VehicleType': 'Hybrid Vehicles', 'Capacity': '90'}, {'VehicleType': 'Trucks', 'Capacity': '50'}, {'VehicleType': 'Sports Cars', 'Capacity': '30'}, {'VehicleType': 'Compact Cars', 'Capacity': '110'}, {'VehicleType': 'Luxury Sedans', 'Capacity': '40'}, {'VehicleType': 'Vans', 'Capacity': '60'}, {'VehicleType': 'Pickup Trucks', 'Capacity': '35'}], 'products': [{'ProductName': 'Sedans', 'Value': '1200'}, {'ProductName': 'SUVs', 'Value': '1800'}, {'ProductName': 'Electric Vehicles', 'Value': '2500'}, {'ProductName': 'Hybrid Vehicles', 'Value': '2000'}, {'ProductName': 'Trucks', 'Value': '1500'}, {'ProductName': 'Sports Cars', 'Value': '3000'}, {'ProductName': 'Compact Cars', 'Value': '1000'}, {'ProductName': 'Luxury Sedans', 'Value': '3500'}, {'ProductName': 'Vans', 'Value': '1600'}, {'ProductName': 'Pickup Trucks', 'Value': '1700'}]}
    vehicle_types = [rec['VehicleType'] for rec in data['capacity']]
    product_names = [rec['ProductName'] for rec in data['products']]
    I = [vt for vt in vehicle_types if vt in product_names]
    b_i = {}
    u_i = {}
    for vt in I:
        prod = next((rec for rec in data['products'] if rec['ProductName'] == vt), None)
        cap = next((rec for rec in data['capacity'] if rec['VehicleType'] == vt), None)
        if prod is None or cap is None:
            raise ValueError(f'Missing data for vehicle type {vt}')
        try:
            b_i[vt] = float(prod['Value'])
            u_i[vt] = float(cap['Capacity'])
        except Exception:
            raise ValueError(f'Non-numeric data for vehicle type {vt}')
    C = sum((u_i[vt] for vt in I))
    m = gp.Model('Car_Inventory_Optimization')
    x = m.addVars(I, lb=0, ub=[u_i[vt] for vt in I], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b_i[vt] * x[vt] for vt in I)), GRB.MAXIMIZE)
    m.addConstrs((x[vt] <= u_i[vt] for vt in I), name='')
    m.addConstr(gp.quicksum((x[vt] for vt in I)) <= C, name='total_capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')