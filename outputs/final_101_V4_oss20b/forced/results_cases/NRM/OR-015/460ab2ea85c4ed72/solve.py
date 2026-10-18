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
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'RestaurantSalesreport.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Aalop',
                                         'format': None,
                                         'inclusive': 'both',
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
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in tbl.get('aliases', []):
        table = tbl
        break
if table is None:
    raise ValueError('Required table not found')
records = table['records']
products = []
revenues = []
demands = []
inventories = []
for rec in records:
    vals = rec['values']
    prod = vals['Product Name']
    rev = float(vals['Revenue'])
    dem = int(vals['Demand'])
    inv = float(vals['Initial Inventory'])
    products.append(prod)
    revenues.append(rev)
    demands.append(dem)
    inventories.append(inv)
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x_vars = {}
for prod, rev, dem, inv in zip(products, revenues, demands, inventories):
    var = m.addVar(vtype=GRB.INTEGER, lb=1)
    x_vars[prod] = var
    m.addConstr(var <= inv)
    m.addConstr(var <= dem)
m.setObjective(gp.quicksum((rev * x_vars[prod] for prod, rev in zip(products, revenues))), GRB.MAXIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for prod, var in x_vars.items():
        print(f'{var.VarName} = {var.X}')
else:
    print('Solver status:', m.status)