CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'aliases': ['file_0_view_0', 'DifferentStoreSales.csv'],
             'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'DifferentStoreSales.csv',
             'filters': {'conditions': [{'column': 'Product_Name',
                                         'dtype': 'string',
                                         'evidence': 'Books',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Books'}],
                         'logic': 'and'},
             'original_rows': 40,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1980',
                                     'Initial Inventory': '9920.0',
                                     'Product_Name': 'Books_15.15',
                                     'Revenue': '15.15'}},
                         {'source_row': 1,
                          'values': {'Demand': '3024',
                                     'Initial Inventory': '20160.0',
                                     'Product_Name': 'Books_30.3',
                                     'Revenue': '30.3'}},
                         {'source_row': 2,
                          'values': {'Demand': '4536',
                                     'Initial Inventory': '30000.0',
                                     'Product_Name': 'Books_45.45',
                                     'Revenue': '45.45'}},
                         {'source_row': 3,
                          'values': {'Demand': '5601',
                                     'Initial Inventory': '38360.0',
                                     'Product_Name': 'Books_60.6',
                                     'Revenue': '60.6'}},
                         {'source_row': 4,
                          'values': {'Demand': '7567',
                                     'Initial Inventory': '51450.0',
                                     'Product_Name': 'Books_75.75',
                                     'Revenue': '75.75'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in t.get('aliases', [])))
records = table['records']
product_names = []
revenues = []
demands = []
inventories = []
for rec in records:
    vals = rec['values']
    product_names.append(vals['Product_Name'])
    revenues.append(float(vals['Revenue']))
    demands.append(int(vals['Demand']))
    inventories.append(float(vals['Initial Inventory']))
m = gp.Model('Books_Revenue_Maximization')
m.setParam('MIPGap', 1e-06)
x = m.addVars(len(product_names), vtype=GRB.INTEGER, lb=0, ub=[min(demands[i], inventories[i]) for i in range(len(product_names))])
m.setObjective(gp.quicksum((revenues[i] * x[i] for i in range(len(product_names)))), GRB.MAXIMIZE)
for i in range(len(product_names)):
    m.addConstr(x[i] <= demands[i])
    m.addConstr(x[i] <= inventories[i])
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'Objective Value: {m.ObjVal}')
    for i in range(len(product_names)):
        var = x[i]
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver Status: {m.Status}')