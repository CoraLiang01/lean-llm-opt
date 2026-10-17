CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A property developer is planning to develop real estate in New York.There are several areas to choose from, '
          'such as Queens and Brooklyn.However, due to limited resources and focus, the developer is not able to '
          'develop all the real estate in all the areas and has to select a few for development.The development '
          'benefit data for real estate in each area is recorded in the ‘products.csv’ file.The developer has an '
          'overall development capacity limit, which is detailed in the ‘capacity.csv’ file.The goal is to decide how '
          'many real estate in each areas are to be developed, in order to maximise the overall benefits while '
          'adhering to the overall development capacity.The decision variable x_i represents the scale of development '
          'per day in area i.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '586'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}},
                         {'source_row': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}},
                         {'source_row': 2, 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}},
                         {'source_row': 3, 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}},
                         {'source_row': 4, 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}},
                         {'source_row': 5, 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}},
                         {'source_row': 8, 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}},
                         {'source_row': 10, 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}},
                         {'source_row': 11, 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}},
                         {'source_row': 12, 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}},
                         {'source_row': 13, 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}},
                         {'source_row': 14, 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}},
                         {'source_row': 15, 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}},
                         {'source_row': 17, 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}},
                         {'source_row': 18, 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}},
                         {'source_row': 19, 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}],
             'returned_rows': 20,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    capacity_table = None
    product_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            capacity_table = tbl
        elif tbl.get('table_id') == 'file_1_view_0':
            product_table = tbl
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA.")
    if product_table is None:
        raise ValueError("Product table 'file_1_view_0' not found in CSVQA_DATA.")
    cap_records = capacity_table.get('records', [])
    if not cap_records:
        raise ValueError('No records found in capacity table.')
    try:
        C = float(cap_records[0]['values']['Capacity'])
    except (KeyError, ValueError, TypeError):
        raise ValueError('Capacity value missing or invalid.')
    product_records = product_table.get('records', [])
    if not product_records:
        raise ValueError('No records found in product table.')
    I = []
    V = {}
    W = {}
    for rec in product_records:
        vals = rec.get('values', {})
        try:
            pname = vals['ProductName']
            v = float(vals['Value'])
            w = float(vals['Weight'])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'Missing or invalid data in product record: {rec}')
        if pname in I:
            raise ValueError(f'Duplicate product name detected: {pname}')
        I.append(pname)
        V[pname] = v
        W[pname] = w
    for pname in I:
        if pname not in V or pname not in W:
            raise ValueError(f'Missing V or W for product {pname}')
    m = gurobipy.Model('RealEstateDevelopment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((V[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((W[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()