CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'An e-commerce platform sells women’s clothing items with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘FAUX’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent '
          'the number of units of each ‘FAUX’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'ZARASales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '‘FAUX’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'FAUX'}],
                         'logic': 'and'},
             'original_rows': 194,
             'records': [{'source_row': 56,
                          'values': {'Demand': '3025',
                                     'Initial Inventory': '20970',
                                     'Product Name': 'FAUX FUR JEWEL SWEATER',
                                     'Revenue': '35.9'}},
                         {'source_row': 57,
                          'values': {'Demand': '9585',
                                     'Initial Inventory': '71970',
                                     'Product Name': 'FAUX LEATHER BOMBER JACKET',
                                     'Revenue': '69.9'}},
                         {'source_row': 58,
                          'values': {'Demand': '4486',
                                     'Initial Inventory': '32730',
                                     'Product Name': 'FAUX LEATHER BOXY FIT JACKET',
                                     'Revenue': '99.9'}},
                         {'source_row': 59,
                          'values': {'Demand': '10322',
                                     'Initial Inventory': '71130',
                                     'Product Name': 'FAUX LEATHER JACKET',
                                     'Revenue': '99.9'}},
                         {'source_row': 60,
                          'values': {'Demand': '4868',
                                     'Initial Inventory': '34910',
                                     'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION',
                                     'Revenue': '159.0'}},
                         {'source_row': 61,
                          'values': {'Demand': '8482',
                                     'Initial Inventory': '64010',
                                     'Product Name': 'FAUX LEATHER PUFFER JACKET',
                                     'Revenue': '69.99'}},
                         {'source_row': 62,
                          'values': {'Demand': '2607',
                                     'Initial Inventory': '20760',
                                     'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS',
                                     'Revenue': '99.9'}},
                         {'source_row': 63,
                          'values': {'Demand': '1784',
                                     'Initial Inventory': '12490',
                                     'Product Name': 'FAUX SHEARLING PLAID JACKET',
                                     'Revenue': '89.9'}},
                         {'source_row': 64,
                          'values': {'Demand': '6626',
                                     'Initial Inventory': '50300',
                                     'Product Name': 'FAUX SUEDE BOMBER JACKET',
                                     'Revenue': '69.9'}},
                         {'source_row': 65,
                          'values': {'Demand': '3256',
                                     'Initial Inventory': '24570',
                                     'Product Name': 'FAUX SUEDE JACKET',
                                     'Revenue': '89.9'}},
                         {'source_row': 66,
                          'values': {'Demand': '2955',
                                     'Initial Inventory': '24430',
                                     'Product Name': 'FAUX SUEDE OVERSHIRT',
                                     'Revenue': '69.9'}},
                         {'source_row': 67,
                          'values': {'Demand': '910',
                                     'Initial Inventory': '7070',
                                     'Product Name': 'FAUX SUEDE PATCH JACKET',
                                     'Revenue': '89.9'}}],
             'returned_rows': 12,
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
    records = table['records']
    I = []
    r = {}
    s = {}
    d = {}
    for rec in records:
        vals = rec['values']
        pname = vals['Product Name']
        if not pname.startswith('FAUX'):
            continue
        try:
            revenue = float(vals['Revenue'])
            initial_inventory = int(vals['Initial Inventory'])
            demand = int(vals['Demand'])
        except Exception as e:
            raise RuntimeError(f'Invalid data for product {pname}: {e}')
        I.append(pname)
        r[pname] = revenue
        s[pname] = initial_inventory
        d[pname] = demand
    if not set(I) == set(r.keys()) == set(s.keys()) == set(d.keys()):
        raise RuntimeError('Mismatch in index sets or missing data for some products.')
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub={i: min(s[i], d[i]) for i in I}, name='')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve()