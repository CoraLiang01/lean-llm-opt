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
                                         'evidence': 'products classified under ‘Organ’',
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
             'role': 'product revenue, demand, and inventory parameters',
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
Revenue = {}
InitialInventory = {}
Demand = {}
for rec in table['records']:
    vals = rec['values']
    subcat = vals['Sub Category']
    if not re.search('Organ', subcat):
        continue
    key = subcat
    try:
        revenue = float(vals['Revenue'])
        initial_inventory = float(vals['Initial Inventory'])
        demand = float(vals['Demand'])
    except Exception as e:
        raise RuntimeError(f'Failed to parse coefficients for {subcat}: {e}')
    I.append(key)
    Revenue[key] = revenue
    InitialInventory[key] = initial_inventory
    Demand[key] = demand
for key in I:
    if key not in Revenue or key not in InitialInventory or key not in Demand:
        raise RuntimeError(f"Missing data for product '{key}'.")
m = Model()
m.Params.MIPGap = 0.0001
x = {}
for i in I:
    ub = min(InitialInventory[i], Demand[i])
    x[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name='x')
m.setObjective(sum((Revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= InitialInventory[i], name='inv')
    m.addConstr(x[i] <= Demand[i], name='dem')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in I:
        print(f'{x[i].VarName}: {x[i].X}')
else:
    print(f'Solver status: {m.Status}')