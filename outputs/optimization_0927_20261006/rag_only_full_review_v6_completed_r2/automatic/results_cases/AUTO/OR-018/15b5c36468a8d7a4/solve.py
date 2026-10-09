CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Baby’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
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
import re
baby_records = []
for rec in table['records']:
    pname = rec['values']['Product Name']
    if re.match('^Baby', pname):
        baby_records.append(rec)
if not baby_records:
    raise RuntimeError("No records found with Product Name prefix 'Baby'.")
I = []
r = {}
d = {}
s = {}
for rec in baby_records:
    pname = rec['values']['Product Name']
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
for pname in I:
    if pname not in r or pname not in d or pname not in s:
        raise RuntimeError(f"Missing parameter for product '{pname}'.")

def build_model():
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub=None, name='')
    m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
    m.addConstrs((x_vars[i] <= s[i] for i in I), name='')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status', m.Status)