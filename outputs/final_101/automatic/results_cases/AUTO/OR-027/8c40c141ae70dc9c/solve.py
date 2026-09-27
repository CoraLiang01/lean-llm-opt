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
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError('Required table file_0_view_0 not found in CSVQA_DATA.')
records = table['records']
I = []
A = {}
d = {}
S = {}
for rec in records:
    vals = rec['values']
    subcat = vals['Sub Category']
    try:
        revenue = float(vals['Revenue'])
        demand = int(float(vals['Demand']))
        inventory = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f"Invalid data for product '{subcat}': {e}")
    I.append(subcat)
    A[subcat] = revenue
    d[subcat] = demand
    S[subcat] = inventory
if not set(A.keys()) == set(d.keys()) == set(S.keys()) == set(I):
    raise ValueError('Mismatch in keys among revenue, demand, inventory, or index set.')
m = gp.Model('Organ_Product_Revenue_Max')
x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((A[i] * x[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= d[i] for i in I), name='')
m.addConstrs((x[i] <= S[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')