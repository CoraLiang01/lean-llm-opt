CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'Baby' product i",
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory',
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
    raise RuntimeError('Required table_id file_0_view_0 not found in CSVQA_DATA.')
I = []
Revenue = {}
InitInv = {}
Demand = {}
prefix = 'Baby'
for rec in table['records']:
    pname = rec['values']['Product Name']
    if not re.match('^Baby', pname):
        continue
    I.append(pname)
    try:
        Revenue[pname] = float(rec['values']['Revenue'])
        InitInv[pname] = float(rec['values']['Initial Inventory'])
        Demand[pname] = float(rec['values']['Demand'])
    except Exception as e:
        raise RuntimeError(f'Error parsing data for product {pname}: {e}')
for pname in I:
    if pname not in Revenue or pname not in InitInv or pname not in Demand:
        raise RuntimeError(f'Missing data for product {pname} in index set I.')
m = Model()
m.Params.MIPGap = 0.0001
x = m.addVars(I, lb=0, ub={i: min(InitInv[i], Demand[i]) for i in I}, vtype=GRB.CONTINUOUS, name='')
m.setObjective(sum((Revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= InitInv[i], name='')
    m.addConstr(x[i] <= Demand[i], name='')
    m.addConstr(x[i] >= 0, name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in x.values():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)