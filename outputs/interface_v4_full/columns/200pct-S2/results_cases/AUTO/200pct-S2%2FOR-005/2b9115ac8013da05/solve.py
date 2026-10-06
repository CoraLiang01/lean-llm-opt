CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of '
          'bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited '
          'storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread '
          'to order each day to maximize our total expected profit while staying within our storage limits. The '
          'decision variables x_i represents the number of units of bread type i to be ordered each day.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['resource_capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'resource_capacity': '180'}}],
             'returned_rows': 1,
             'role': 'storage capacity constraint',
             'table_id': 'file_0_view_0'},
            {'columns': ['item_name', 'item_value', 'resource_requirement'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'item_name': 'Baguette', 'item_value': '888', 'resource_requirement': '4'}},
                         {'source_row': 1,
                          'values': {'item_name': 'Croissant', 'item_value': '134', 'resource_requirement': '2'}},
                         {'source_row': 2,
                          'values': {'item_name': 'Sourdough', 'item_value': '129', 'resource_requirement': '4'}},
                         {'source_row': 3,
                          'values': {'item_name': 'Rye Bread', 'item_value': '370', 'resource_requirement': '3'}},
                         {'source_row': 4,
                          'values': {'item_name': 'Brioche', 'item_value': '921', 'resource_requirement': '2'}},
                         {'source_row': 5,
                          'values': {'item_name': 'Focaccia', 'item_value': '765', 'resource_requirement': '1'}},
                         {'source_row': 6,
                          'values': {'item_name': 'Ciabatta', 'item_value': '154', 'resource_requirement': '2'}},
                         {'source_row': 7,
                          'values': {'item_name': 'Pita', 'item_value': '837', 'resource_requirement': '1'}},
                         {'source_row': 8,
                          'values': {'item_name': 'Bagel', 'item_value': '584', 'resource_requirement': '3'}},
                         {'source_row': 9,
                          'values': {'item_name': 'English Muffin', 'item_value': '365', 'resource_requirement': '3'}}],
             'returned_rows': 10,
             'role': 'products and profit coefficients',
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
    a = {}
    for rec in products_table['records']:
        name = rec['values']['item_name']
        I.append(name)
        try:
            v[name] = float(rec['values']['item_value'])
            a[name] = float(rec['values']['resource_requirement'])
        except Exception as e:
            raise ValueError(f'Invalid data for item {name}: {e}')
    try:
        C = float(capacity_table['records'][0]['values']['resource_capacity'])
    except Exception as e:
        raise ValueError(f'Invalid capacity data: {e}')
    m = gp.Model('Bakery_Bread_Order')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((a[i] * x[i] for i in I)) <= C, name='storage_capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')