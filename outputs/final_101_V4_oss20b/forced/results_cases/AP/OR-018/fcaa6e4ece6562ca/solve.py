CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'aliases': ['file_0_view_0', 'Salesdata.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'inclusive': 'both',
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
table = None
for t in CSVQA_DATA['tables']:
    if t.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in t.get('aliases', []):
        table = t
        break
records = table['records']
r = []
d = []
I = []
for rec in records:
    vals = rec['values']
    r.append(float(vals['Revenue']))
    d.append(int(vals['Demand']))
    I.append(int(vals['Initial Inventory']))
m = gp.Model('Baby_Product_Sales')
x = m.addVars(len(r), vtype=GRB.INTEGER)
m.setObjective(gp.quicksum((r[i] * x[i] for i in range(len(r)))), GRB.MAXIMIZE)
for i in range(len(r)):
    m.addConstr(x[i] <= d[i])
    m.addConstr(x[i] <= I[i])
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective Value:', m.ObjVal)
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print('Status:', m.Status)