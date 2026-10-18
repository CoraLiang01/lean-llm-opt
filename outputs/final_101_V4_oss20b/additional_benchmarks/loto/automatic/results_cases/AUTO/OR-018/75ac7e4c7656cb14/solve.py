CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'Salesdata.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = CSVQA_DATA['tables'][0]
records = table['records']
m = gp.Model('BabyProductRevenueMaximization')
vars = {}
for rec in records:
    vals = rec['values']
    product_name = vals['Product Name']
    revenue = float(vals['Revenue'])
    demand = int(vals['Demand'])
    inventory = int(vals['Initial Inventory'])
    x = m.addVar(vtype=GRB.INTEGER, lb=0)
    vars[product_name] = x
    m.addConstr(x <= inventory)
    m.addConstr(x <= demand)
m.setObjective(gp.quicksum((revenue * vars[product_name] for product_name, revenue in [(rec['values']['Product Name'], float(rec['values']['Revenue'])) for rec in records])), GRB.MAXIMIZE)
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective Value:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver Status:', m.Status)