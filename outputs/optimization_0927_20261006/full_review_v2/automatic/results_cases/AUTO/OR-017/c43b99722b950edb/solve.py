CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailStoreSalesTransactions(ScannerData).csv',
             'filters': {'conditions': [{'column': 'SKU',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘ZZ’',
                                         'operator': 'contains',
                                         'value': 'ZZ'}],
                         'logic': 'or'},
             'original_rows': 5242,
             'records': [{'source_row': 387,
                          'values': {'Demand': '12', 'Initial Inventory': '100.0', 'Revenue': '37.58', 'SKU': '2L6ZZ'}},
                         {'source_row': 449,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '8.56', 'SKU': '2ZZWJ'}},
                         {'source_row': 939,
                          'values': {'Demand': '40', 'Initial Inventory': '220.0', 'Revenue': '13.59', 'SKU': '6BZZ3'}},
                         {'source_row': 1854,
                          'values': {'Demand': '133', 'Initial Inventory': '670.0', 'Revenue': '4.71', 'SKU': 'CHZZA'}},
                         {'source_row': 2082,
                          'values': {'Demand': '180',
                                     'Initial Inventory': '1090.0',
                                     'Revenue': '22.3',
                                     'SKU': 'DZZBG'}},
                         {'source_row': 2634,
                          'values': {'Demand': '86', 'Initial Inventory': '430.0', 'Revenue': '2.0', 'SKU': 'HZZ6U'}},
                         {'source_row': 2997,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '2.55', 'SKU': 'KFUZZ'}},
                         {'source_row': 3231,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '25.62', 'SKU': 'LWDZZ'}},
                         {'source_row': 3697,
                          'values': {'Demand': '40', 'Initial Inventory': '230.0', 'Revenue': '42.87', 'SKU': 'P62ZZ'}},
                         {'source_row': 4027,
                          'values': {'Demand': '22', 'Initial Inventory': '110.0', 'Revenue': '12.93', 'SKU': 'RDOZZ'}},
                         {'source_row': 4124,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '1.58', 'SKU': 'RZZTA'}},
                         {'source_row': 4739,
                          'values': {'Demand': '37', 'Initial Inventory': '190.0', 'Revenue': '20.75', 'SKU': 'WEXZZ'}},
                         {'source_row': 4826,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '31.99', 'SKU': 'WZZJ8'}},
                         {'source_row': 5237,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}},
                         {'source_row': 5238,
                          'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}},
                         {'source_row': 5239,
                          'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}},
                         {'source_row': 5240,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}},
                         {'source_row': 5241,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}],
             'returned_rows': 18,
             'role': 'product revenue, demand, and inventory for ZZ products',
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
I = []
A = {}
d = {}
I_init = {}
for rec in records:
    vals = rec['values']
    sku = vals['SKU']
    if 'ZZ' not in sku:
        continue
    I.append(sku)
    try:
        A[sku] = float(vals['Revenue'])
        d[sku] = int(vals['Demand'])
        I_init[sku] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Error parsing numeric fields for SKU {sku}: {e}')
for sku in I:
    if sku not in A or sku not in d or sku not in I_init:
        raise ValueError(f'Missing parameter for SKU {sku}')

def build_model(I, A, d, I_init):
    m = gp.Model('Retail_ZZ_Rev_Max')
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
    m.addConstrs((x_vars[i] <= I_init[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    return m
m = build_model(I, A, d, I_init)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')