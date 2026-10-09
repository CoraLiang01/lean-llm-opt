CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement '
          'for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock '
          'capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day '
          'and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall '
          'stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered '
          'each day.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '875'}}],
             'returned_rows': 1,
             'role': 'overall stock capacity constraint',
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
             'role': 'product income statement',
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
products = []
value = {}
weight = {}
for rec in products_table['records']:
    pname = rec['values']['ProductName']
    try:
        v = int(rec['values']['Value'])
        w = int(rec['values']['Weight'])
    except Exception:
        raise ValueError(f'Non-integer Value or Weight for product {pname}')
    products.append(pname)
    value[pname] = v
    weight[pname] = w
if len(capacity_table['records']) != 1:
    raise ValueError('Capacity table must have exactly one record.')
try:
    C = int(capacity_table['records'][0]['values']['Capacity'])
except Exception:
    raise ValueError('Non-integer Capacity in capacity table.')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Mismatch in product indices for value/weight.')
m = gp.Model('Supermarket_Stock_Replenishment')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in products)) <= C, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')