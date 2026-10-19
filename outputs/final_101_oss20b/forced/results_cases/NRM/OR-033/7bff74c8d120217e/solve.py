CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', 'EuropeSalesRecords.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'classified under ‘Baby’',
                                         'inclusive': 'both',
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
data = CSVQA_DATA
table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
product_names = []
revenues = []
demands = []
inventories = []
for rec in records:
    vals = rec['values']
    product_names.append(vals['Product Name'])
    revenues.append(float(vals['Revenue']))
    demands.append(int(vals['Demand']))
    inventories.append(int(vals['Initial Inventory']))
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x_vars = []
for i, pname in enumerate(product_names):
    var = m.addVar(vtype=GRB.INTEGER, lb=0)
    x_vars.append(var)
for i, var in enumerate(x_vars):
    m.addConstr(var <= demands[i])
    m.addConstr(var <= inventories[i])
m.setObjective(gp.quicksum((revenues[i] * x_vars[i] for i in range(len(x_vars)))), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.Status)