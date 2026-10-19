CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'aliases': ['file_0_view_0', 'EuropeSalesRecords.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']
table = next((t for t in tables if t.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in t.get('aliases', [])))
records = table['records']
product_names = []
revenues = []
demands = []
inventories = []
for rec in records:
    vals = rec['values']
    product_names.append(vals['Product Name'])
    revenues.append(float(vals['Revenue']))
    demands.append(int(vals['Demand']))
    inventories.append(int(vals['Initial Inventory']))
m = gp.Model('Baby_Food_Revenue_Maximization')
m.setParam('MIPGap', 1e-06)
x = m.addVars(len(product_names), vtype=GRB.INTEGER)
m.setObjective(gp.quicksum((revenues[i] * x[i] for i in range(len(product_names)))), GRB.MAXIMIZE)
for i in range(len(product_names)):
    m.addConstr(x[i] <= demands[i])
    m.addConstr(x[i] <= inventories[i])
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective Value:', m.ObjVal)
    for i in range(len(product_names)):
        print(f'{x[i].VarName} = {x[i].X}')
else:
    print('Solver Status:', m.Status)