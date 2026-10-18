CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'SupermartGrocerySales-RetailAnalyticsDataset.csv'],
             'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'filters': {'conditions': [{'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': 'classified under ‘Organ’',
                                         'operator': 'contains',
                                         'value': 'Organ'}],
                         'logic': 'and'},
             'original_rows': 23,
             'records': [{'source_row': 17,
                          'values': {'Demand': '678906',
                                     'Initial Inventory': '5034020.0',
                                     'Revenue': '60.8',
                                     'Sub Category': 'Organic Fruits'}},
                         {'source_row': 18,
                          'values': {'Demand': '749927',
                                     'Initial Inventory': '5589290.0',
                                     'Revenue': '918.45',
                                     'Sub Category': 'Organic Staples'}},
                         {'source_row': 19,
                          'values': {'Demand': '699808',
                                     'Initial Inventory': '5202710.0',
                                     'Revenue': '77.52',
                                     'Sub Category': 'Organic Vegetables'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0':
        table = tbl
        break
if table is None:
    raise ValueError('Table file_0_view_0 not found')
records = table['records']
items = []
revenue = []
demand = []
initial_inventory = []
for rec in records:
    vals = rec['values']
    item = vals['Sub Category']
    rev = float(vals['Revenue'])
    dem = int(vals['Demand'])
    inv = float(vals['Initial Inventory'])
    items.append(item)
    revenue.append(rev)
    demand.append(dem)
    initial_inventory.append(inv)
m = gp.Model('max_revenue')
m.Params.MIPGap = 1e-06
x_vars = {}
for i, item in enumerate(items):
    var = m.addVar(vtype=GRB.INTEGER)
    x_vars[item] = var
    m.addConstr(var <= initial_inventory[i])
    m.addConstr(var <= demand[i])
m.setObjective(gp.quicksum((revenue[i] * x_vars[items[i]] for i in range(len(items)))), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective Value:', m.ObjVal)
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print('Status:', m.Status)