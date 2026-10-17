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
 'route': 'AP',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '4466'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    cap_table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if cap_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA")
    if not cap_table.get('records'):
        raise ValueError('Capacity table has no records')
    cap_record = cap_table['records'][0]
    try:
        C = float(cap_record['values']['Capacity'])
    except (KeyError, ValueError, TypeError):
        raise ValueError('Capacity value missing or invalid')
    prod_table = next((t for t in tables if t.get('table_id') == 'file_1_view_0'), None)
    if prod_table is None:
        raise ValueError("Products table 'file_1_view_0' not found in CSVQA_DATA")
    records = prod_table.get('records', [])
    if not records:
        raise ValueError('Products table has no records')
    b = {}
    w = {}
    for rec in records:
        vals = rec.get('values', {})
        prod = vals.get('ProductName')
        if prod is None:
            raise ValueError(f'Missing ProductName in record {rec}')
        try:
            b_val = float(vals['Value'])
            w_val = float(vals['Weight'])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'Missing or invalid Value/Weight for product {prod}')
        if prod in b or prod in w:
            raise ValueError(f'Duplicate product key {prod}')
        b[prod] = b_val
        w[prod] = w_val
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(b.keys(), vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((b[i] * x[i] for i in b)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w[i] * x[i] for i in w)) <= C)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()