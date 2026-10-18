CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'EuropeSalesRecords.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
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
    raise ValueError('Required table not found in CSVQA_DATA')
records = table['records']
products = []
profits = {}
demands = {}
inventories = {}
for rec in records:
    vals = rec['values']
    prod = vals['Product Name']
    rev = float(vals['Revenue'])
    dem = int(vals['Demand'])
    inv = int(vals['Initial Inventory'])
    products.append(prod)
    profits[prod] = rev
    demands[prod] = dem
    inventories[prod] = inv
m = gp.Model('BabyProductRevenueMaximization')
m.setParam(GRB.Param.MIPGap, 1e-06)
vars = {}
for prod in products:
    vars[prod] = m.addVar(vtype=GRB.INTEGER, lb=0)
for prod in products:
    m.addConstr(vars[prod] <= demands[prod])
    m.addConstr(vars[prod] <= inventories[prod])
m.setObjective(gp.quicksum((profits[prod] * vars[prod] for prod in products)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective Value:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver Status:', m.Status)