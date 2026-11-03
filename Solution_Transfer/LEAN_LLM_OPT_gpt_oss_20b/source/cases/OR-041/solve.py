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
 'route': 'RA',
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
    capacity_table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA")
    capacity_records = capacity_table.get('records', [])
    if not capacity_records:
        raise ValueError('No records found in capacity table')
    capacity_value_str = capacity_records[0].get('values', {}).get('Capacity')
    if capacity_value_str is None:
        raise ValueError('Capacity value missing in capacity table')
    try:
        capacity = int(capacity_value_str)
    except ValueError:
        raise ValueError(f"Capacity value '{capacity_value_str}' is not an integer")
    products_table = next((t for t in tables if t.get('table_id') == 'file_1_view_0'), None)
    if products_table is None:
        raise ValueError("Products table 'file_1_view_0' not found in CSVQA_DATA")
    product_records = products_table.get('records', [])
    if not product_records:
        raise ValueError('No records found in products table')
    product_names = []
    value_dict = {}
    weight_dict = {}
    for rec in product_records:
        vals = rec.get('values', {})
        name = vals.get('ProductName')
        if name is None:
            raise ValueError('ProductName missing in a product record')
        value_str = vals.get('Value')
        weight_str = vals.get('Weight')
        if value_str is None or weight_str is None:
            raise ValueError(f"Value or Weight missing for product '{name}'")
        try:
            value = int(value_str)
            weight = int(weight_str)
        except ValueError:
            raise ValueError(f"Non-integer Value or Weight for product '{name}'")
        product_names.append(name)
        value_dict[name] = value
        weight_dict[name] = weight
    for name in product_names:
        if name not in value_dict or name not in weight_dict:
            raise ValueError(f"Missing Value or Weight for product '{name}'")
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value_dict[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight_dict[i] * x[i] for i in product_names)) <= capacity, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()