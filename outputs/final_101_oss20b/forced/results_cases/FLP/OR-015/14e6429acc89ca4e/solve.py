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
 'route': 'FLP',
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
records = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0' or 'file_0_view_0' in tbl.get('aliases', []):
        records = tbl['records']
        break
if records is None:
    raise ValueError('Required table not found')
products = []
for rec in records:
    vals = rec['values']
    name = vals['Product Name']
    revenue = float(vals['Revenue'])
    demand = float(vals['Demand'])
    inventory = float(vals['Initial Inventory'])
    products.append((name, revenue, demand, inventory))
m = gp.Model('Aalop_Sales')
m.setParam('MIPGap', 1e-06)
x = {}
for name, revenue, demand, inventory in products:
    x[name] = m.addVar(lb=0)
m.setObjective(gp.quicksum((revenue * x[name] for name, revenue, demand, inventory in products)), GRB.MAXIMIZE)
for name, revenue, demand, inventory in products:
    m.addConstr(x[name] <= demand)
    m.addConstr(x[name] <= inventory)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.status}')