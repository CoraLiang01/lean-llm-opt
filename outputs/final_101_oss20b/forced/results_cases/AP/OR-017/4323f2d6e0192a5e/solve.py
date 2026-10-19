CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'aliases': ['file_0_view_0', 'RetailStoreSalesTransactions(ScannerData).csv'],
             'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailStoreSalesTransactions(ScannerData).csv',
             'filters': {'conditions': [{'column': 'SKU',
                                         'dtype': 'string',
                                         'evidence': 'classified under ‘ZZ’',
                                         'operator': 'contains',
                                         'value': 'ZZ'}],
                         'logic': 'and'},
             'original_rows': 5242,
             'records': [{'source_row': 5237,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}},
                         {'source_row': 5238,
                          'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}},
                         {'source_row': 5239,
                          'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}},
                         {'source_row': 5240,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}},
                         {'source_row': 5241,
                          'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in t.get('aliases', [])))
records = table['records']
skus = []
revs = []
dems = []
invs = []
for rec in records:
    vals = rec['values']
    skus.append(vals['SKU'])
    revs.append(float(vals['Revenue']))
    dems.append(int(vals['Demand']))
    invs.append(float(vals['Initial Inventory']))
n = len(skus)
m = gp.Model('ZZ_Product_Optimization')
x = m.addVars(n, vtype=GRB.INTEGER)
m.setObjective(gp.quicksum((revs[i] * x[i] for i in range(n))), GRB.MAXIMIZE)
for i in range(n):
    m.addConstr(x[i] <= dems[i])
    m.addConstr(x[i] <= invs[i])
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'Objective Value: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} = {v.X}')
else:
    print(f'Status: {m.Status}')