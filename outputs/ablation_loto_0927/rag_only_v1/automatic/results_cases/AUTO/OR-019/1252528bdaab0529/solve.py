CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '‘27in’ products',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
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
             'role': 'revenue management products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB
data = CSVQA_DATA
table = None
for t in data['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
I = []
r = {}
d = {}
s = {}
prefix = '27in'
for rec in table['records']:
    pname = rec['values']['Product Name']
    if not re.match('^27in', pname):
        continue
    I.append(pname)
    try:
        r[pname] = float(rec['values']['Revenue'])
        d[pname] = int(rec['values']['Demand'])
        s[pname] = int(rec['values']['Initial Inventory'])
    except Exception as e:
        raise RuntimeError(f"Error parsing parameters for product '{pname}': {e}")
for pname in I:
    if pname not in r or pname not in d or pname not in s:
        raise RuntimeError(f"Missing parameter for product '{pname}'.")
m = Model()
m.setParam('MIPGap', 0.0001)
x = {}
for i in I:
    ub = min(d[i], s[i])
    x[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name='x')
m.setObjective(sum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= d[i], name='demand')
    m.addConstr(x[i] <= s[i], name='inventory')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for i in I:
        print(x[i].VarName, x[i].X)
else:
    print('Solver status:', m.Status)