CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to restock its inventory, and for each type of produce (such as leafy vegetables, '
          'mushrooms, etc.), there is an associated benefit table provided in "products.csv." The supermarket faces an '
          'overall inventory-capacity constraint, provided in “capacity.csv.”. The goal is to decide the daily order '
          'quantity of each produce type so as to maximize total benefit while ensuring that the total weight of all '
          'ordered units does not exceed the overall capacity.The decision variables x_i represents the number of '
          'units of type i of produce to be ordered daily.The decision variables must be integers.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', 'capacity.csv'],
             'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '1035'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'products.csv'],
             'columns': ['ProductName', 'Weight', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
capacity_table = None
for table in CSVQA_DATA['tables']:
    if table['table_id'] == 'file_0_view_0' or 'capacity.csv' in table.get('aliases', []):
        capacity_table = table
        break
if capacity_table is None:
    raise ValueError('Capacity table not found')
capacity = int(capacity_table['records'][0]['values']['Capacity'])
products_table = None
for table in CSVQA_DATA['tables']:
    if table['table_id'] == 'file_1_view_0' or 'products.csv' in table.get('aliases', []):
        products_table = table
        break
if products_table is None:
    raise ValueError('Products table not found')
products = []
for rec in products_table['records']:
    prod_name = rec['values']['ProductName']
    weight = float(rec['values']['Weight'])
    value = float(rec['values']['Value'])
    products.append((prod_name, weight, value))
m = gp.Model('produce_order')
m.setParam('MIPGap', 1e-06)
vars = {}
for prod_name, weight, value in products:
    var_name = f'x_{prod_name.replace(' ', '_')}'
    vars[prod_name] = m.addVar(vtype=GRB.INTEGER)
m.update()
m.setObjective(gp.quicksum((vars[prod_name] * value for prod_name, _, value in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((vars[prod_name] * weight for prod_name, weight, _ in products)) <= capacity, 'capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for prod_name in vars:
        var = vars[prod_name]
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Status: {m.Status}')