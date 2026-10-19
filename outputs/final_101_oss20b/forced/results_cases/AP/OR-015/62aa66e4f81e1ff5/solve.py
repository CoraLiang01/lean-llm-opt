CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for '
          'these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant '
          'aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, '
          'which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, '
          'and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to '
          'be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The '
          'variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to '
          'fulfill.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'aliases': ['file_0_view_0', 'RestaurantSalesreport.csv'],
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Aalop',
                                         'operator': 'contains',
                                         'value': 'Aalop'}],
                         'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1483',
                                     'Initial Inventory': '10440.0',
                                     'Product Name': 'Aalopuri',
                                     'Revenue': '20'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
records = CSVQA_DATA['tables'][0]['records']
products = []
revenues = []
demands = []
inventories = []
for rec in records:
    vals = rec['values']
    products.append(vals['Product Name'])
    revenues.append(int(vals['Revenue']))
    demands.append(int(vals['Demand']))
    inventories.append(float(vals['Initial Inventory']))
m = gp.Model('RestaurantSalesOptimization')
m.setParam('MIPGap', 1e-06)
x = {}
for i, prod in enumerate(products):
    var = m.addVar(vtype=GRB.INTEGER)
    x[i] = var
m.setObjective(gp.quicksum((revenues[i] * x[i] for i in range(len(products)))), GRB.MAXIMIZE)
for i in range(len(products)):
    m.addConstr(x[i] <= inventories[i])
    m.addConstr(x[i] <= demands[i])
    m.addConstr(x[i] >= 0)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective Value:', m.ObjVal)
    for i in range(len(products)):
        print(f'{x[i].VarName} = {x[i].X}')
else:
    print('Solver Status:', m.Status)