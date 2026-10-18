CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'aliases': ['file_0_view_0', 'SalesDataAnalysis.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
products = []
revenues = {}
demands = {}
inventories = {}
for rec in records:
    prod = rec['values']['Product Name']
    prod_key = prod.replace(' ', '_')
    products.append(prod_key)
    revenues[prod_key] = float(rec['values']['Revenue'])
    demands[prod_key] = int(rec['values']['Demand'])
    inventories[prod_key] = int(rec['values']['Initial Inventory'])
m = gp.Model('Maximize_Revenue')
x = m.addVars(products, lb=0)
m.setObjective(gp.quicksum((revenues[p] * x[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= demands[p])
    m.addConstr(x[p] <= inventories[p])
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status:', m.Status)