CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
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
import gurobipy
from gurobipy import GRB
tables = CSVQA_DATA.get('tables', [])
table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
if table is None:
    raise ValueError('Required table file_0_view_0 not found in CSVQA_DATA')
records = table.get('records', [])
I = []
r_i = {}
d_i = {}
s_i = {}
for rec in records:
    vals = rec.get('values', {})
    prod = vals.get('Product Name')
    if prod is None:
        continue
    I.append(prod)
    rev_str = vals.get('Revenue')
    if rev_str is None:
        raise ValueError(f'Missing Revenue for product {prod}')
    r_i[prod] = float(rev_str)
    dem_str = vals.get('Demand')
    if dem_str is None:
        raise ValueError(f'Missing Demand for product {prod}')
    d_i[prod] = int(dem_str)
    inv_str = vals.get('Initial Inventory')
    if inv_str is None:
        raise ValueError(f'Missing Initial Inventory for product {prod}')
    s_i[prod] = int(inv_str)
m = gurobipy.Model('27in_product_sales')
m.setParam('MIPGap', 1e-06)
x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= d_i[i] for i in I), name='')
m.addConstrs((x[i] <= s_i[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')