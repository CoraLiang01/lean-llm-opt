CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'SalesDataAnalysis.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'and'},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
import gurobipy as gp
from gurobipy import GRB
table = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in tbl.get('aliases', []):
        table = tbl
        break
if table is None:
    raise ValueError('Required table not found')
records = table['records']
m = gp.Model()
m.setParam(GRB.Param.MIPGap, 1e-06)
vars = {}
for idx, rec in enumerate(records):
    vals = rec['values']
    prod_name = vals['Product Name']
    var_name = re.sub('\\W+', '_', prod_name.strip())
    var = m.addVar(vtype=GRB.INTEGER, lb=0)
    vars[prod_name] = var
for prod_name, var in vars.items():
    vals = next((r['values'] for r in records if r['values']['Product Name'] == prod_name))
    demand = int(vals['Demand'])
    inventory = int(vals['Initial Inventory'])
    m.addConstr(var <= demand)
    m.addConstr(var <= inventory)
obj = gp.quicksum((float(rec['values']['Revenue']) * vars[rec['values']['Product Name']] for rec in records))
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(GRB.Status[m.Status])