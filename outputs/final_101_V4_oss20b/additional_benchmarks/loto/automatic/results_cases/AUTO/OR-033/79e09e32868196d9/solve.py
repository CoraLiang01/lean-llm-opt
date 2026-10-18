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
table_id = 'file_0_view_0'
records = CSVQA_DATA['tables'][0]['records']
m = gp.Model('BabyProductRevenueMaximization')
m.setParam('MIPGap', 1e-06)
vars = {}
for rec in records:
    vals = rec['values']
    prod = vals['Product Name']
    rev = float(vals['Revenue'])
    dem = int(vals['Demand'])
    inv = int(vals['Initial Inventory'])
    var_name = f"x_{prod.replace(' ', '_')}"
    x = m.addVar(vtype=GRB.INTEGER, lb=0)
    vars[prod] = (x, rev, dem, inv)
m.update()
m.setObjective(gp.quicksum((vars[p][1] * vars[p][0] for p in vars)), GRB.MAXIMIZE)
for prod, (x, rev, dem, inv) in vars.items():
    m.addConstr(x <= dem)
    m.addConstr(x <= inv)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'Objective Value: {m.objVal}')
    for prod, (x, rev, dem, inv) in vars.items():
        print(f'{x.VarName} = {x.X}')
else:
    print(f'Solver Status: {m.status}')