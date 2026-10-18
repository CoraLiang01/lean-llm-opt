CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in '
          'the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to '
          'maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory '
          'levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The decision variables x_i represent the '
          'number of units of each ‘ZZ’ product i that the store plans to fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'RetailStoreSalesTransactions(ScannerData).csv'],
             'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailStoreSalesTransactions(ScannerData).csv',
             'filters': {'conditions': [{'column': 'SKU',
                                         'dtype': 'string',
                                         'evidence': 'ZZ',
                                         'operator': 'prefix',
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
table = CSVQA_DATA['tables'][0]
records = table['records']
m = gp.Model()
m.setParam('MIPGap', 1e-06)
vars = {}
for rec in records:
    sku = rec['values']['SKU']
    var = m.addVar(vtype=GRB.INTEGER, lb=0)
    vars[sku] = var
m.update()
obj = gp.quicksum((float(rec['values']['Revenue']) * vars[rec['values']['SKU']] for rec in records))
m.setObjective(obj, GRB.MAXIMIZE)
for rec in records:
    sku = rec['values']['SKU']
    demand = int(rec['values']['Demand'])
    inventory = float(rec['values']['Initial Inventory'])
    m.addConstr(vars[sku] <= demand)
    m.addConstr(vars[sku] <= inventory)
m.optimize()
if m.status == GRB.OPTIMAL:
    print('ObjVal:', m.objVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status:', m.status)