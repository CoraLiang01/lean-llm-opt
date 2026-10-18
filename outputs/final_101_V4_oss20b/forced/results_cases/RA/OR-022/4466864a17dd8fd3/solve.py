CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'Salesorders.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '261.2933'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965'}}],
             'returned_rows': 2,
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
    raise ValueError('Required table not found')
records = table['records']
products = []
revenues = {}
demands = {}
inventories = {}
for rec in records:
    vals = rec['values']
    prod = vals['Product Name']
    rev = float(vals['Revenue'])
    dem = int(vals['Demand'])
    inv = int(vals['Initial Inventory'])
    products.append(prod)
    revenues[prod] = rev
    demands[prod] = dem
    inventories[prod] = inv
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x = {}
for prod in products:
    var_name = f'x_{prod.replace(' ', '_')}'
    x[prod] = m.addVar(vtype=GRB.INTEGER, lb=0)
m.update()
for prod in products:
    m.addConstr(x[prod] <= demands[prod])
    m.addConstr(x[prod] <= inventories[prod])
m.setObjective(gp.quicksum((revenues[prod] * x[prod] for prod in products)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for prod in products:
        var = x[prod]
        print(f'{var.VarName} {var.X}')
else:
    print(m.Status)