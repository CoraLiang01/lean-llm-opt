CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of '
          'bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited '
          'storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread '
          'to order each day to maximize our total expected profit while staying within our storage limits. The '
          'decision variables x_i represents the number of units of bread type i to be ordered each day.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '180'}}],
             'returned_rows': 1,
             'role': 'storage capacity',
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
             'role': 'bread products',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
products_table = None
capacity_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_1_view_0':
        products_table = t
    elif t['table_id'] == 'file_0_view_0':
        capacity_table = t
if products_table is None or capacity_table is None:
    raise ValueError('Required tables not found in CSVQA_DATA.')
bread_types = []
profit = {}
weight = {}
for rec in products_table['records']:
    name = rec['values']['ProductName']
    bread_types.append(name)
    try:
        profit[name] = float(rec['values']['Value'])
        weight[name] = float(rec['values']['Weight'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in products.csv for {name}: {e}')
if len(capacity_table['records']) != 1:
    raise ValueError('capacity.csv must have exactly one row.')
try:
    capacity = float(capacity_table['records'][0]['values']['Capacity'])
except Exception as e:
    raise ValueError(f'Invalid numeric value in capacity.csv: {e}')
if set(profit.keys()) != set(bread_types) or set(weight.keys()) != set(bread_types):
    raise ValueError('Mismatch in bread types and parameter keys.')
m = gp.Model('Bakery_Bread_Stocking')
x_vars = m.addVars(bread_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x_vars[i] for i in bread_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in bread_types)) <= capacity, name='storage_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')