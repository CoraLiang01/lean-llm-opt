CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement '
          'for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock '
          'capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day '
          'and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall '
          'stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered '
          'each day.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '875'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Weight', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Spinach', 'Value': '64', 'Weight': '230'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '75', 'Weight': '637'}},
                         {'source_row': 2, 'values': {'ProductName': 'Apples', 'Value': '68', 'Weight': '773'}},
                         {'source_row': 3, 'values': {'ProductName': 'Carrots', 'Value': '11', 'Weight': '653'}},
                         {'source_row': 4, 'values': {'ProductName': 'Basil', 'Value': '91', 'Weight': '755'}},
                         {'source_row': 5, 'values': {'ProductName': 'Potatoes', 'Value': '31', 'Weight': '670'}},
                         {'source_row': 6, 'values': {'ProductName': 'Green Beans', 'Value': '90', 'Weight': '505'}},
                         {'source_row': 7, 'values': {'ProductName': 'Blueberries', 'Value': '56', 'Weight': '821'}},
                         {'source_row': 8, 'values': {'ProductName': 'Oranges', 'Value': '10', 'Weight': '83'}},
                         {'source_row': 9, 'values': {'ProductName': 'Watermelons', 'Value': '24', 'Weight': '249'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            capacity_table = tbl
            break
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA")
    if not capacity_table['records']:
        raise ValueError('Capacity table contains no records')
    capacity_value = capacity_table['records'][0]['values']['Capacity']
    try:
        C = int(capacity_value)
    except Exception as e:
        raise ValueError(f'Invalid capacity value: {capacity_value}') from e
    product_table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_1_view_0':
            product_table = tbl
            break
    if product_table is None:
        raise ValueError("Product table 'file_1_view_0' not found in CSVQA_DATA")
    if not product_table['records']:
        raise ValueError('Product table contains no records')
    I = []
    w = {}
    v = {}
    for rec in product_table['records']:
        key = rec['source_row']
        I.append(key)
        try:
            w[key] = int(rec['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid weight for product {key}') from e
        try:
            v[key] = int(rec['values']['Value'])
        except Exception as e:
            raise ValueError(f'Invalid value for product {key}') from e
    for key in I:
        if key not in w or key not in v:
            raise ValueError(f'Missing parameters for product {key}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w[i] * x[i] for i in I)) <= C, name='Capacity')
    m.optimize()
    return m
m = solve_problem()