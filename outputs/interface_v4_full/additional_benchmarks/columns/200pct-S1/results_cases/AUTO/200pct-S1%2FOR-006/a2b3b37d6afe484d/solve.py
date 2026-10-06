CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'An automobile sales company is planning its inventory replenishment strategy. For each type of vehicle, the '
          'company has a "products.csv" file that records the benefit coefficients for each vehicle type. The company '
          'also has an overall inventory capacity constraint, which is detailed in the "capacity.csv" file. The '
          "company's objective is to decide which vehicle types to order each day and in what quantities to maximize "
          'the overall benefit while adhering to the total inventory capacity. The decision variables x_i represent '
          'the number of units of vehicle type i to be ordered daily. The company needs to balance between maximizing '
          'benefits and complying with inventory constraints to develop the optimal ordering plan.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '1576'}}],
             'returned_rows': 1,
             'role': 'overall inventory capacity constraint',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 25,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Sedan', 'Value': '1752', 'Weight': '15'}},
                         {'source_row': 1, 'values': {'ProductName': 'SUV', 'Value': '1856', 'Weight': '87'}},
                         {'source_row': 2, 'values': {'ProductName': 'Truck', 'Value': '8372', 'Weight': '36'}},
                         {'source_row': 3, 'values': {'ProductName': 'Convertible', 'Value': '6168', 'Weight': '30'}},
                         {'source_row': 4, 'values': {'ProductName': 'Minivan', 'Value': '9681', 'Weight': '33'}},
                         {'source_row': 5, 'values': {'ProductName': 'Coupe', 'Value': '8062', 'Weight': '72'}},
                         {'source_row': 6, 'values': {'ProductName': 'Hatchback', 'Value': '3895', 'Weight': '75'}},
                         {'source_row': 7, 'values': {'ProductName': 'Station Wagon', 'Value': '3254', 'Weight': '71'}},
                         {'source_row': 8, 'values': {'ProductName': 'Electric Car', 'Value': '1701', 'Weight': '51'}},
                         {'source_row': 9, 'values': {'ProductName': 'Hybrid Car', 'Value': '6799', 'Weight': '21'}},
                         {'source_row': 10, 'values': {'ProductName': 'Luxury Sedan', 'Value': '2724', 'Weight': '97'}},
                         {'source_row': 11, 'values': {'ProductName': 'Sports Car', 'Value': '6304', 'Weight': '52'}},
                         {'source_row': 12, 'values': {'ProductName': 'Crossover', 'Value': '3255', 'Weight': '25'}},
                         {'source_row': 13, 'values': {'ProductName': 'Diesel Truck', 'Value': '1923', 'Weight': '15'}},
                         {'source_row': 14, 'values': {'ProductName': 'Compact SUV', 'Value': '4103', 'Weight': '54'}},
                         {'source_row': 15, 'values': {'ProductName': 'Luxury SUV', 'Value': '4429', 'Weight': '57'}},
                         {'source_row': 16, 'values': {'ProductName': 'Cargo Van', 'Value': '2663', 'Weight': '18'}},
                         {'source_row': 17, 'values': {'ProductName': 'Pickup Truck', 'Value': '1691', 'Weight': '69'}},
                         {'source_row': 18, 'values': {'ProductName': 'Roadster', 'Value': '5632', 'Weight': '26'}},
                         {'source_row': 19, 'values': {'ProductName': 'Muscle Car', 'Value': '4793', 'Weight': '38'}},
                         {'source_row': 20,
                          'values': {'ProductName': 'Off-road Vehicle', 'Value': '1343', 'Weight': '31'}},
                         {'source_row': 21, 'values': {'ProductName': 'Camper Van', 'Value': '9124', 'Weight': '74'}},
                         {'source_row': 22, 'values': {'ProductName': 'Compact Car', 'Value': '3652', 'Weight': '82'}},
                         {'source_row': 23, 'values': {'ProductName': 'Motorcycle', 'Value': '8842', 'Weight': '49'}},
                         {'source_row': 24,
                          'values': {'ProductName': 'Electric SUV', 'Value': '9176', 'Weight': '64'}}],
             'returned_rows': 25,
             'role': 'vehicle types and benefit coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    products_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_1_view_0':
            products_table = t
            break
    if products_table is None:
        raise ValueError('Products table not found.')
    capacity_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            capacity_table = t
            break
    if capacity_table is None:
        raise ValueError('Capacity table not found.')
    I = []
    v = {}
    w = {}
    for rec in products_table['records']:
        pname = rec['values']['ProductName']
        I.append(pname)
        try:
            v[pname] = float(rec['values']['Value'])
            w[pname] = float(rec['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid Value or Weight for product {pname}: {e}')
    if len(capacity_table['records']) != 1:
        raise ValueError('Capacity table must have exactly one record.')
    try:
        C = float(capacity_table['records'][0]['values']['Capacity'])
    except Exception as e:
        raise ValueError(f'Invalid Capacity value: {e}')
    if set(I) != set(v.keys()) or set(I) != set(w.keys()):
        raise ValueError('Mismatch in product identifiers for v or w.')
    m = gp.Model('vehicle_inventory')
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
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