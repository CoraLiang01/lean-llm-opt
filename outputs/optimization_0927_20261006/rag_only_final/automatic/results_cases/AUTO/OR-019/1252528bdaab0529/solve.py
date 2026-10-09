CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'27in' products",
                                         'operator': 'prefix',
                                         'value': '27in'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'27in' products",
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'or'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}}],
             'returned_rows': 2,
             'role': 'product revenue, demand, and inventory for 27in products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB, quicksum
tables = CSVQA_DATA['tables']
table = None
for t in tables:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
I = []
r_i = {}
d_i = {}
s_i = {}
for rec in table['records']:
    pname = rec['values']['Product Name']
    if '27in' in pname:
        I.append(pname)
        try:
            r_i[pname] = float(rec['values']['Revenue'])
            d_i[pname] = float(rec['values']['Demand'])
            s_i[pname] = float(rec['values']['Initial Inventory'])
        except Exception as e:
            raise RuntimeError(f"Failed to parse numeric fields for product '{pname}': {e}")
for pname in I:
    if pname not in r_i or pname not in d_i or pname not in s_i:
        raise RuntimeError(f"Missing data for product '{pname}' in index set I.")
m = Model()
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
for pname in I:
    m.addConstr(x_vars[pname] <= s_i[pname], name='inv_' + str(pname))
for pname in I:
    m.addConstr(x_vars[pname] <= d_i[pname], name='dem_' + str(pname))
m.setObjective(quicksum((r_i[pname] * x_vars[pname] for pname in I)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for pname in I:
        print(x_vars[pname].VarName, x_vars[pname].X)
else:
    print('Solver status:', m.Status)