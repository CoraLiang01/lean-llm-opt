CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'RA',
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
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0' or 'file_0_view_0' in t.get('aliases', []):
        table = t
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
    inv = int(vals['Initial Inventory'])
    products.append(prod)
    revenues.append(rev)
    demands.append(dem)
    inventories.append(inv)
m = gp.Model('maximize_revenue')
m.setParam('MIPGap', 1e-06)
vars = {}
for prod, rev, dem, inv in zip(products, revenues, demands, inventories):
    var_name = f"x_{prod.replace(' ', '_')}"
    v = m.addVar(vtype=GRB.INTEGER, lb=0)
    vars[prod] = v
    m.addConstr(v <= dem)
    m.addConstr(v <= inv)
m.update()
obj = gp.quicksum((rev * vars[prod] for prod, rev in zip(products, revenues)))
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(m.objVal)
    for prod, v in vars.items():
        print(v.VarName, v.X)
else:
    print(m.status)