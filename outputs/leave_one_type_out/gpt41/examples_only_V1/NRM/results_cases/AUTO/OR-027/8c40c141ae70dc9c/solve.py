CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'filters': {'conditions': [{'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': "'Organ' product i",
                                         'operator': 'prefix',
                                         'value': 'Organ'},
                                        {'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': "'Organ' product i",
                                         'operator': 'contains',
                                         'value': 'Organ'}],
                         'logic': 'or'},
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
             'role': 'revenue, demand, and inventory for Organ products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
I = []
params = {}
for rec in table['records']:
    subcat = rec['values']['Sub Category']
    if re.search('Organ', subcat):
        key = subcat
        try:
            r = float(rec['values']['Revenue'])
            s = float(rec['values']['Initial Inventory'])
            d = float(rec['values']['Demand'])
        except Exception as e:
            raise RuntimeError(f"Invalid numeric data for product '{subcat}': {e}")
        params[subcat] = {'r': r, 's': s, 'd': d}
        I.append(subcat)
for i in I:
    if i not in params or not all((k in params[i] for k in ('r', 's', 'd'))):
        raise RuntimeError(f"Missing parameter(s) for product '{i}'.")
m = Model()
m.Params.MIPGap = 0.0001
x = m.addVars(I, vtype=GRB.INTEGER, lb=0, obj=0, name='')
m.setObjective(sum((params[i]['r'] * x[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= params[i]['s'], name='')
    m.addConstr(x[i] <= params[i]['d'], name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for i in I:
        print(x[i].VarName, x[i].X)
else:
    print('Solver status:', m.Status)