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
from gurobipy import Model, GRB, quicksum

def build_and_solve():
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
        if not prod.startswith('Baby'):
            continue
        I.append(prod)
        try:
            r[prod] = float(vals['Revenue'])
            d[prod] = int(vals['Demand'])
            s[prod] = int(vals['Initial Inventory'])
        except Exception as e:
            raise RuntimeError(f"Error parsing numeric fields for product '{prod}': {e}")
    if not I:
        raise RuntimeError("No products with prefix 'Baby' found in table_id 'file_0_view_0'.")
    for prod in I:
        if prod not in r or prod not in d or prod not in s:
            raise RuntimeError(f"Missing parameter for product '{prod}'.")
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    for i in I:
        m.addConstr(x_vars[i] <= d[i], name='demand')
        m.addConstr(x_vars[i] <= s[i], name='inventory')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in I:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve()