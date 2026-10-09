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
                                         'evidence': 'products classified under ‘27in’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': '27in'}],
                         'logic': 'or'},
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
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table['records']
items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    pname = vals['Product Name']
    if not pname.startswith('27in'):
        continue
    try:
        rev = float(vals['Revenue'])
        dem = int(vals['Demand'])
        inv = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Invalid data for product '{pname}': {e}")
    items.append(pname)
    revenue[pname] = rev
    demand[pname] = dem
    inventory[pname] = inv
for i in items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f"Missing data for product '{i}'.")

def build_model():
    m = gp.Model('27in_Product_Revenue_Maximization')
    x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')