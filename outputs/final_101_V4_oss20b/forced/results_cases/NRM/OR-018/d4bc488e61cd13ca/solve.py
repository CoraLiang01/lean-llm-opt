CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'Salesdata.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in tbl.get('aliases', []):
        table = tbl
        break
if table is None:
    raise ValueError('Table file_0_view_0 not found')
records = table['records']
m = gp.Model('BabyProductsRevenueMaximization')
m.setParam(GRB.Param.MIPGap, 1e-06)
vars = {}
for idx, rec in enumerate(records):
    prod_name = rec['values']['Product Name']
    revenue = float(rec['values']['Revenue'])
    demand = int(rec['values']['Demand'])
    inventory = int(rec['values']['Initial Inventory'])
    var_name = f'x_{idx}'
    var = m.addVar(vtype=GRB.INTEGER, lb=0)
    vars[idx] = {'var': var, 'revenue': revenue, 'demand': demand, 'inventory': inventory}
m.update()
obj_expr = gp.quicksum((vars[i]['revenue'] * vars[i]['var'] for i in vars))
m.setObjective(obj_expr, GRB.MAXIMIZE)
for i in vars:
    var = vars[i]['var']
    m.addConstr(var <= vars[i]['demand'])
    m.addConstr(var <= vars[i]['inventory'])
m.optimize()
if m.status == GRB.OPTIMAL:
    print(m.objVal)
    for i in vars:
        var = vars[i]['var']
        print(f'{var.VarName} {var.X}')
else:
    print(m.status)