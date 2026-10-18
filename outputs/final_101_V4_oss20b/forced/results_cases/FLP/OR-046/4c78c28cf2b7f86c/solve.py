CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement '
          'for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock '
          'capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day '
          'and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall '
          'stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered '
          'each day.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'aliases': ['file_0_view_0', 'capacity.csv'],
             'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '875'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'products.csv'],
             'columns': ['ProductName', 'Weight', 'Value'],
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
import gurobipy as gp
from gurobipy import GRB
capacity_table = CSVQA_DATA['tables'][0]
C = float(capacity_table['records'][0]['values']['Capacity'])
products_table = CSVQA_DATA['tables'][1]
n = len(products_table['records'])
v = []
w = []
for rec in products_table['records']:
    v.append(float(rec['values']['Value']))
    w.append(float(rec['values']['Weight']))
m = gp.Model('Product_Selection')
m.setParam('MIPGap', 1e-06)
x = m.addVars(n, lb=0)
m.setObjective(gp.quicksum((v[i] * x[i] for i in range(n))), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[i] * x[i] for i in range(n))) <= C)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for i in range(n):
        print(f'{x[i].VarName} = {x[i].X}')
else:
    print('Status:', m.status)