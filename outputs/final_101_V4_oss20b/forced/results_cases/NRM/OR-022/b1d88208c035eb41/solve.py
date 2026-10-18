CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'Salesorders.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
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
data = CSVQA_DATA
table = None
for tbl in data['tables']:
    if tbl.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in tbl.get('aliases', []):
        table = tbl
        break
if table is None:
    raise ValueError('Required table not found')
records = table['records']
m = gp.Model()
m.setParam(GRB.Param.MIPGap, 1e-06)
vars = []
for idx, rec in enumerate(records):
    w = float(rec['values']['Revenue'])
    d = float(rec['values']['Demand'])
    c = float(rec['values']['Initial Inventory'])
    var = m.addVar(vtype=GRB.INTEGER)
    vars.append((var, w, d, c))
for var, w, d, c in vars:
    m.addConstr(var <= c)
    m.addConstr(var <= d)
m.setObjective(gp.quicksum((w * var for var, w, d, c in vars)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var, _, _, _ in vars:
        print(f'{var.VarName} {var.X}')
else:
    print(m.Status)