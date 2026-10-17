CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of '
          'bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited '
          'storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread '
          'to order each day to maximize our total expected profit while staying within our storage limits. The '
          'decision variables x_i represents the number of units of bread type i to be ordered each day.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '180'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}},
                         {'source_row': 1, 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}},
                         {'source_row': 2, 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}},
                         {'source_row': 3, 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}},
                         {'source_row': 4, 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}},
                         {'source_row': 5, 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}},
                         {'source_row': 6, 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}},
                         {'source_row': 7, 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}},
                         {'source_row': 8, 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}},
                         {'source_row': 9, 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    capacity_table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA.")
    capacity_records = capacity_table.get('records', [])
    if len(capacity_records) != 1:
        raise ValueError('Capacity table must contain exactly one record.')
    try:
        capacity = int(capacity_records[0]['values']['Capacity'])
    except (KeyError, ValueError, TypeError):
        raise ValueError('Invalid capacity value.')
    if capacity < 0:
        raise ValueError('Capacity must be non-negative.')
    products_table = next((t for t in tables if t.get('table_id') == 'file_1_view_0'), None)
    if products_table is None:
        raise ValueError("Products table 'file_1_view_0' not found in CSVQA_DATA.")
    product_records = products_table.get('records', [])
    if not product_records:
        raise ValueError('Products table must contain at least one record.')
    product_names = []
    profit = {}
    weight = {}
    for rec in product_records:
        try:
            name = rec['values']['ProductName']
            val = int(rec['values']['Value'])
            wt = int(rec['values']['Weight'])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'Invalid product record: {rec}')
        if name in product_names:
            raise ValueError(f'Duplicate product name detected: {name}')
        product_names.append(name)
        profit[name] = val
        weight[name] = wt
    m = gurobipy.Model('Bakery_Profit_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((profit[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight[i] * x[i] for i in product_names)) <= capacity, name='Capacity')
    m.optimize()
    return m
m = solve_problem()