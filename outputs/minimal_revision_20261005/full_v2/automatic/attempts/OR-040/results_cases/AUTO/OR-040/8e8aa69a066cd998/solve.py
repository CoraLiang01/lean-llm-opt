CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A real-estate developer is planning property development in New York City. For each area (e.g., Queens, '
          'Brooklyn, etc.), the developer has a “products.csv” file that records the benefit coefficient for that '
          'area.\n'
          '    The developer also faces a single overall development-capacity constraint, provided in “capacity.csv.” '
          'file. The objective is to decide the daily scale of development in each area so as to maximize total '
          'benefit while ensuring that the sum of all development units does not exceed the overall capacity. The '
          'decision variables  x_i  represent the scale of development in area  i  each day.The decision variables '
          'must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '4466'}}],
             'returned_rows': 1,
             'role': 'overall development capacity constraint',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '443', 'Weight': '104'}},
                         {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '522', 'Weight': '368'}},
                         {'source_row': 2, 'values': {'ProductName': 'Manhattan', 'Value': '300', 'Weight': '483'}},
                         {'source_row': 3, 'values': {'ProductName': 'Bronx', 'Value': '767', 'Weight': '165'}},
                         {'source_row': 4, 'values': {'ProductName': 'Staten Island', 'Value': '300', 'Weight': '105'}},
                         {'source_row': 5, 'values': {'ProductName': 'Harlem', 'Value': '309', 'Weight': '123'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Upper East Side', 'Value': '598', 'Weight': '131'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Lower Manhattan', 'Value': '460', 'Weight': '341'}},
                         {'source_row': 8, 'values': {'ProductName': 'Midtown', 'Value': '318', 'Weight': '258'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Long Island City', 'Value': '126', 'Weight': '469'}},
                         {'source_row': 10, 'values': {'ProductName': 'Williamsburg', 'Value': '593', 'Weight': '387'}},
                         {'source_row': 11, 'values': {'ProductName': 'Bushwick', 'Value': '871', 'Weight': '425'}},
                         {'source_row': 12, 'values': {'ProductName': 'Flatbush', 'Value': '858', 'Weight': '482'}},
                         {'source_row': 13, 'values': {'ProductName': 'Greenpoint', 'Value': '321', 'Weight': '495'}},
                         {'source_row': 14, 'values': {'ProductName': 'Park Slope', 'Value': '275', 'Weight': '305'}},
                         {'source_row': 15, 'values': {'ProductName': 'Astoria', 'Value': '700', 'Weight': '377'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Jackson Heights', 'Value': '685', 'Weight': '318'}},
                         {'source_row': 17, 'values': {'ProductName': 'Flushing', 'Value': '940', 'Weight': '56'}},
                         {'source_row': 18, 'values': {'ProductName': 'Sunnyside', 'Value': '522', 'Weight': '213'}},
                         {'source_row': 19, 'values': {'ProductName': 'Ditmars', 'Value': '763', 'Weight': '472'}}],
             'returned_rows': 20,
             'role': 'area decision and benefit coefficients',
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
        raise ValueError('Products table not found')
    capacity_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            capacity_table = t
            break
    if capacity_table is None:
        raise ValueError('Capacity table not found')
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
            raise ValueError(f'Invalid Value/Weight for {pname}: {e}')
    if len(capacity_table['records']) != 1:
        raise ValueError('Capacity table must have exactly one record')
    try:
        C = float(capacity_table['records'][0]['values']['Capacity'])
    except Exception as e:
        raise ValueError(f'Invalid Capacity: {e}')
    if set(v.keys()) != set(I) or set(w.keys()) != set(I):
        raise ValueError('Mismatch in product keys for v or w')
    m = gp.Model('NYC_Development')
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