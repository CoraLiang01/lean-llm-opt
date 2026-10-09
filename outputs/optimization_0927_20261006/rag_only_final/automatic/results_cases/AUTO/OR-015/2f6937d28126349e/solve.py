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
                                         'evidence': '‘Aalop’',
                                         'inclusive': 'both',
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
             'role': 'Aalop product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB, quicksum
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
I = []
r = {}
d = {}
s = {}
for rec in table['records']:
    vals = rec['values']
    prod = vals['Product Name']
    try:
        revenue = float(vals['Revenue'])
        demand = int(vals['Demand'])
        inventory = float(vals['Initial Inventory'])
    except Exception as e:
        raise RuntimeError(f"Error parsing numeric fields for product '{prod}': {e}")
    I.append(prod)
    r[prod] = revenue
    d[prod] = demand
    s[prod] = inventory
if set(r.keys()) != set(I) or set(d.keys()) != set(I) or set(s.keys()) != set(I):
    raise RuntimeError('Parameter keys do not match index set I.')
m = Model()
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub={i: min(d[i], s[i]) for i in I}, name='')
for i in I:
    m.addConstr(x_vars[i] <= s[i], name='')
    m.addConstr(x_vars[i] <= d[i], name='')
m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in I:
        print(f'{x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')