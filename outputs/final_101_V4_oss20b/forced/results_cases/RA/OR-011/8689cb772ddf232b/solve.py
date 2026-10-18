CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the '
          '‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective '
          'is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. '
          'The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. '
          'During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n'
          '\n'
          'Demand for each product during the sales period is assumed to be deterministic and known in advance, with '
          'demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of '
          'units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative '
          'integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the '
          'fulfillment quantities must satisfy both inventory and demand constraints.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'OnlineRetailSalesDataset.csv'],
             'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': 'id999',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'id999'}],
                         'logic': 'and'},
             'original_rows': 900,
             'records': [{'source_row': 899,
                          'values': {'Demand': '8171',
                                     'Initial Inventory': '56450',
                                     'Revenue': '434.74',
                                     'id_number': 'id999'}}],
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
    raise ValueError('Required table not found')
records = table['records']
m = gp.Model()
m.setParam('MIPGap', 1e-06)
vars = {}
rec_dict = {}
for rec in records:
    id_num = rec['values']['id_number']
    rec_dict[id_num] = rec
    rev = float(rec['values']['Revenue'])
    demand = int(rec['values']['Demand'])
    inventory = int(rec['values']['Initial Inventory'])
    x = m.addVar(vtype=GRB.INTEGER)
    vars[id_num] = x
    m.addConstr(x <= inventory)
    m.addConstr(x <= demand)
m.setObjective(gp.quicksum((vars[id_num] * float(rec_dict[id_num]['values']['Revenue']) for id_num in vars)), GRB.MAXIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for id_num, var in vars.items():
        print(f'{var.VarName} = {var.X}')
else:
    print('Status:', m.status)