CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'TP',
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
table = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in tbl.get('aliases', []):
        table = tbl
        break
if table is None:
    raise ValueError('Required table not found')
records = table['records']
indices = []
revenues = {}
demands = {}
inventories = {}
names = {}
for idx, rec in enumerate(records, start=1):
    prod_name = rec['values']['Product Name']
    revenue = float(rec['values']['Revenue'])
    demand = float(rec['values']['Demand'])
    inventory = float(rec['values']['Initial Inventory'])
    indices.append(idx)
    revenues[idx] = revenue
    demands[idx] = demand
    inventories[idx] = inventory
    names[idx] = prod_name
m = gp.Model('27in_product_sales')
x = {}
for i in indices:
    var_name = f'x_{i}'
    x[i] = m.addVar(vtype=GRB.CONTINUOUS)
m.update()
m.setObjective(gp.quicksum((revenues[i] * x[i] for i in indices)), GRB.MAXIMIZE)
for i in indices:
    m.addConstr(x[i] <= demands[i])
    m.addConstr(x[i] <= inventories[i])
m.setParam(GRB.Param.MIPGap, 1e-06)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'Objective value: {m.objVal}')
    for i in indices:
        var = x[i]
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Status: {m.status}')