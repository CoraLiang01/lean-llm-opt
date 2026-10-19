CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'TP',
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
m = gp.Model()
table = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in tbl.get('aliases', []):
        table = tbl
        break
if table is None:
    raise ValueError('Table file_0_view_0 not found')
records = table['records']
vars_dict = {}
for rec in records:
    vals = rec['values']
    prod_name = vals['Product Name']
    revenue = float(vals['Revenue'])
    demand = float(vals['Demand'])
    init_inv = float(vals['Initial Inventory'])
    var_name = f'x_{prod_name}'
    var = m.addVar(vtype=GRB.CONTINUOUS)
    vars_dict[prod_name] = (var, revenue, demand, init_inv)
m.update()
obj = gp.quicksum((vars_dict[prod][1] * vars_dict[prod][0] for prod in vars_dict))
m.setObjective(obj, GRB.MAXIMIZE)
for prod in vars_dict:
    var, revenue, demand, init_inv = vars_dict[prod]
    m.addConstr(var >= demand)
    m.addConstr(var <= init_inv)
m.setParam(GRB.Param.MIPGap, 1e-06)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} = {v.X}')
else:
    print(f'Status: {m.status}')