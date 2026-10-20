CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for '
          'these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant '
          'aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, '
          'which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, '
          'and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to '
          'be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The '
          'variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to '
          'fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'RestaurantSalesreport.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Aalop',
                                         'operator': 'contains',
                                         'value': 'Aalop'}],
                         'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1483',
                                     'Initial Inventory': '10440.0',
                                     'Product Name': 'Aalopuri',
                                     'Revenue': '20'}}],
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
if table is None:
    raise ValueError('Table not found')
records = table['records']
p = {}
d = {}
s = {}
for rec in records:
    vals = rec['values']
    prod = vals['Product Name']
    p[prod] = float(vals['Revenue'])
    d[prod] = int(vals['Demand'])
    s[prod] = float(vals['Initial Inventory'])
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x = {}
for prod in p:
    x[prod] = m.addVar(vtype=GRB.INTEGER, lb=0)
m.update()
for prod in p:
    m.addConstr(x[prod] <= d[prod])
    m.addConstr(x[prod] <= s[prod])
m.setObjective(gp.quicksum((p[prod] * x[prod] for prod in p)), GRB.MAXIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(m.ObjVal)
    for prod in p:
        var = x[prod]
        print(f'{var.VarName} {var.X}')
else:
    print(m.status)