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
                                         'evidence': 'products classified under ‘Baby’',
                                         'inclusive': 'both',
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
records = []
for rec in table['records']:
    pname = rec['values']['Product Name']
    if pname.startswith('Baby'):
        records.append(rec)
if not records:
    raise RuntimeError("No records found with Product Name prefix 'Baby'.")
I = []
r = {}
d = {}
s = {}
for rec in records:
    pname = rec['values']['Product Name']
    try:
        revenue = float(rec['values']['Revenue'])
        demand = int(float(rec['values']['Demand']))
        inventory = int(float(rec['values']['Initial Inventory']))
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
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub={i: min(d[i], s[i]) for i in I}, name='')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')