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
table = [r['values'] for r in CSVQA_DATA['tables'][0]['records']]
if not table:
    raise ValueError("No records found for 'Baby' products in file_0_view_0.")
I = []
revenue = {}
demand = {}
inventory = {}
for row in table:
    product = row['Product Name']
    try:
        A_i = float(row['Revenue'])
        d_i = int(row['Demand'])
        I_i = int(row['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {product}: {e}')
    I.append(product)
    revenue[product] = A_i
    demand[product] = d_i
    inventory[product] = I_i
for product in I:
    if product not in revenue or product not in demand or product not in inventory:
        raise ValueError(f'Missing data for product {product}.')
m = gp.Model('Baby_Product_Revenue_Maximization')
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x_vars[i] <= min(inventory[i], demand[i]), name='ub_' + str(i))
    m.addConstr(x_vars[i] >= 0, name='lb_' + str(i))
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')