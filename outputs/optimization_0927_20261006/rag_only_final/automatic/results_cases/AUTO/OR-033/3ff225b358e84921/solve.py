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
from gurobipy import Model, GRB, quicksum
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
I = []
r = {}
d = {}
s = {}
for rec in table['records']:
    pname = rec['values']['Product Name']
    if not pname.startswith('Baby'):
        continue
    I.append(pname)
    try:
        r[pname] = float(rec['values']['Revenue'])
        d[pname] = float(rec['values']['Demand'])
        s[pname] = float(rec['values']['Initial Inventory'])
    except Exception as e:
        raise RuntimeError(f"Error parsing numeric fields for product '{pname}': {e}")
if set(I) != set(r.keys()) or set(I) != set(d.keys()) or set(I) != set(s.keys()):
    raise RuntimeError('Mismatch in index set and parameter keys for r, d, or s.')
m = Model()
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
for i in I:
    m.addConstr(x_vars[i] <= s[i], name='')
    m.addConstr(x_vars[i] <= d[i], name='')
m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in I:
        print(f'{x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')