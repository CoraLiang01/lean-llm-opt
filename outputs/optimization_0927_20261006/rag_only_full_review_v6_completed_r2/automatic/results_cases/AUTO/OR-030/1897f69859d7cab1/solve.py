CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ '
          'column. The dealership aims to maximize total revenue using the initial inventory of car models classified '
          'under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are '
          'specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'BigMartSales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '‘FDK57’',
                                         'operator': 'exact',
                                         'value': 'FDK57'}],
                         'logic': 'and'},
             'original_rows': 5681,
             'records': [{'source_row': 1163,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1501,
                          'values': {'Demand': '40',
                                     'Initial Inventory': '100',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1576,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 1793,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.144'}},
                         {'source_row': 2438,
                          'values': {'Demand': '10',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.544'}},
                         {'source_row': 4297,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 4806,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '250',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.744'}},
                         {'source_row': 4942,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.844'}}],
             'returned_rows': 8,
             'role': 'car model revenue, demand, and inventory',
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
Revenue = {}
Inventory = {}
Demand = {}
for rec in table['records']:
    vals = rec['values']
    if vals['Product Name'] != 'FDK57':
        continue
    key = rec['source_row']
    try:
        Revenue[key] = float(vals['Revenue'])
        Inventory[key] = int(vals['Initial Inventory'])
        Demand[key] = int(vals['Demand'])
    except Exception as e:
        raise RuntimeError(f'Invalid data in record {rec}: {e}')
    I.append(key)
if not set(I) == set(Revenue) == set(Inventory) == set(Demand):
    raise RuntimeError('Mismatch in index sets or missing data for some car models.')
m = Model()
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub={i: min(Inventory[i], Demand[i]) for i in I}, name='')
m.setObjective(quicksum((Revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x_vars[i] <= Inventory[i], name='')
    m.addConstr(x_vars[i] <= Demand[i], name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for i in I:
        print(x_vars[i].VarName, x_vars[i].X)
else:
    print(m.Status)