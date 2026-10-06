CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'27in' product i",
                                         'inclusive': 'both',
                                         'operator': 'prefix',
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
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB
data = CSVQA_DATA
table_id = 'file_0_view_0'
table = None
for t in data['tables']:
    if t['table_id'] == table_id:
        table = t
        break
if table is None:
    raise RuntimeError('Required table_id not found in CSVQA_DATA.')
I = []
r = {}
d = {}
s = {}
for rec in table['records']:
    pname = rec['values']['Product Name']
    if not re.match('^27in', pname):
        continue
    try:
        revenue = float(rec['values']['Revenue'])
        demand = int(rec['values']['Demand'])
        inventory = int(rec['values']['Initial Inventory'])
    except Exception as e:
        raise RuntimeError(f"Invalid data for product '{pname}': {e}")
    I.append(pname)
    r[pname] = revenue
    d[pname] = demand
    s[pname] = inventory
for i in I:
    if i not in r or i not in d or i not in s:
        raise RuntimeError(f"Missing data for product '{i}'.")
m = Model()
m.Params.MIPGap = 0.0001
x = m.addVars(I, vtype=GRB.INTEGER, lb=0, obj=0, name='')
m.setObjective(sum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= s[i], name='')
    m.addConstr(x[i] <= d[i], name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for i in I:
        print(x[i].VarName, x[i].X)
else:
    print('Status', m.Status)