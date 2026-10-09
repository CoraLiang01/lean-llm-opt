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
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '"Baby"',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product demand, revenue, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError('Required table_id file_0_view_0 not found in CSVQA_DATA.')
records = table['records']
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    prod = vals['Product Name']
    try:
        rev = float(vals['Revenue'])
        dem = int(vals['Demand'])
        inv = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {prod}: {e}')
    products.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv
for prod in products:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f'Missing parameter for product {prod}')
m = gp.Model('Baby_Product_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[prod] * x_vars[prod] for prod in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[prod] <= inventory[prod] for prod in products), name='')
m.addConstrs((x_vars[prod] <= demand[prod] for prod in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')