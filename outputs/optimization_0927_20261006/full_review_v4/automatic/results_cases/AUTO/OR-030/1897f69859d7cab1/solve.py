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
if not records:
    raise ValueError("No records found for 'FDK57' in file_0_view_0.")
I = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    idx = rec['source_row']
    try:
        A_i = float(vals['Revenue'])
        d_i = int(vals['Demand'])
        s_i = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data in record {rec}: {e}')
    I.append(idx)
    revenue[idx] = A_i
    demand[idx] = d_i
    inventory[idx] = s_i
if not set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(I):
    raise ValueError('Mismatch in index sets for parameters.')
m = gp.Model('Car_Dealership_FDK57')
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in I), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')