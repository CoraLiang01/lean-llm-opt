CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ '
          'column. The dealership aims to maximize total revenue using the initial inventory of car models classified '
          'under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are '
          'specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'BigMartSales.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'BigMartSales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'FDK57',
                                         'operator': 'contains',
                                         'value': 'FDK57'}],
                         'logic': 'and'},
             'original_rows': 5681,
             'records': [{'source_row': 1163,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1501,
                          'values': {'Demand': '40',
                                     'Initial Inventory': '100',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1576,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 1793,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.144'}},
                         {'source_row': 2438,
                          'values': {'Demand': '10',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.544'}},
                         {'source_row': 4297,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 4806,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '250',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.744'}},
                         {'source_row': 4942,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.844'}}],
             'returned_rows': 8,
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
m = gp.Model()
m.setParam('MIPGap', 1e-06)
vars = []
for idx, rec in enumerate(records):
    w = float(rec['values']['Revenue'])
    I = int(rec['values']['Initial Inventory'])
    d = int(rec['values']['Demand'])
    var = m.addVar(vtype=GRB.INTEGER, lb=1)
    vars.append((var, w, I, d))
m.update()
m.setObjective(gp.quicksum((var * w for var, w, _, _ in vars)), GRB.MAXIMIZE)
for var, _, I, d in vars:
    m.addConstr(var <= I)
    m.addConstr(var <= d)
m.addConstr(gp.quicksum((var for var, _, _, _ in vars)) <= 100)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for var, _, _, _ in vars:
        print(f'{var.VarName} = {var.X}')
else:
    print('Status:', m.status)