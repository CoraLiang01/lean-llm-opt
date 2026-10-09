CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to restock its inventory, and for each type of produce (such as leafy vegetables, '
          'mushrooms, etc.), there is an associated benefit table provided in "products.csv." The supermarket faces an '
          'overall inventory-capacity constraint, provided in “capacity.csv.”. The goal is to decide the daily order '
          'quantity of each produce type so as to maximize total benefit while ensuring that the total weight of all '
          'ordered units does not exceed the overall capacity.The decision variables x_i represents the number of '
          'units of type i of produce to be ordered daily.The decision variables must be integers.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '1035'}}],
             'returned_rows': 1,
             'role': 'overall inventory-capacity constraint',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Weight', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Spinach', 'Value': '49', 'Weight': '282'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '30', 'Weight': '83'}},
                         {'source_row': 2, 'values': {'ProductName': 'Apples', 'Value': '30', 'Weight': '251'}},
                         {'source_row': 3, 'values': {'ProductName': 'Carrots', 'Value': '18', 'Weight': '257'}},
                         {'source_row': 4, 'values': {'ProductName': 'Basil', 'Value': '54', 'Weight': '88'}},
                         {'source_row': 5, 'values': {'ProductName': 'Potatoes', 'Value': '27', 'Weight': '52'}},
                         {'source_row': 6, 'values': {'ProductName': 'Green Beans', 'Value': '91', 'Weight': '198'}},
                         {'source_row': 7, 'values': {'ProductName': 'Blueberries', 'Value': '88', 'Weight': '203'}},
                         {'source_row': 8, 'values': {'ProductName': 'Oranges', 'Value': '78', 'Weight': '87'}},
                         {'source_row': 9, 'values': {'ProductName': 'Watermelons', 'Value': '22', 'Weight': '265'}}],
             'returned_rows': 10,
             'role': 'produce benefit and weight table',
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
values = {}
weights = {}
for rec in products_table['records']:
    pname = rec['values']['ProductName']
    try:
        v = int(rec['values']['Value'])
        w = int(rec['values']['Weight'])
    except Exception:
        raise ValueError(f'Non-integer Value or Weight for product {pname}')
    products.append(pname)
    values[pname] = v
    weights[pname] = w
if len(capacity_table['records']) != 1:
    raise ValueError('Expected exactly one capacity record.')
try:
    capacity = int(capacity_table['records'][0]['values']['Capacity'])
except Exception:
    raise ValueError('Non-integer Capacity value.')
if set(values.keys()) != set(products) or set(weights.keys()) != set(products):
    raise ValueError('Mismatch in product indices for values/weights.')
m = gp.Model('Supermarket_Inventory_Restock')
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x_vars[i] for i in products)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')