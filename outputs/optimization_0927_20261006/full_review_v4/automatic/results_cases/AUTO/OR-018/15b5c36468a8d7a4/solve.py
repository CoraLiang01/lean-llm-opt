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
                                         'evidence': 'products classified under ‘Baby’',
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
             'role': 'descriptive non-unique role',
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
    raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table['records']
if not records:
    raise ValueError("No records found for products classified under 'Baby'.")
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    pname = vals['Product Name']
    try:
        rev = float(vals['Revenue'])
        dem = int(vals['Demand'])
        inv = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Invalid data for product '{pname}': {e}")
    products.append(pname)
    revenue[pname] = rev
    demand[pname] = dem
    inventory[pname] = inv
for pname in products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing data for product '{pname}'.")
m = gp.Model('Baby_Product_Revenue_Maximization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')