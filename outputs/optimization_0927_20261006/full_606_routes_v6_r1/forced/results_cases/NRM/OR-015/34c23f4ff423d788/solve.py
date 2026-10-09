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
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'classified under ‘Aalop’',
                                         'operator': 'prefix',
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
table = CSVQA_DATA['tables'][0]
records = table['records']
items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    product = vals['Product Name']
    items.append(product)
    try:
        revenue[product] = float(vals['Revenue'])
        demand[product] = int(vals['Demand'])
        inventory[product] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {product}: {e}')
if not set(items) == set(revenue) == set(demand) == set(inventory):
    raise ValueError('Mismatch in index sets for items, revenue, demand, or inventory.')
m = gp.Model('Aalop_Inventory_Optimization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')