CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '‘27in’ product',
                                         'operator': 'prefix',
                                         'value': '27in'}],
                         'logic': 'or'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '261.2933'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB, quicksum

def build_and_solve(CSVQA_DATA):
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise RuntimeError('Required table_id file_0_view_0 not found in CSVQA_DATA.')
    I = []
    r = {}
    s = {}
    d = {}
    for rec in table['records']:
        pname = rec['values']['Product Name']
        if not pname.startswith('27in'):
            continue
        I.append(pname)
        try:
            r[pname] = float(rec['values']['Revenue'])
            s[pname] = float(rec['values']['Initial Inventory'])
            d[pname] = float(rec['values']['Demand'])
        except Exception as e:
            raise RuntimeError(f"Failed to parse numeric fields for product '{pname}': {e}")
    for pname in I:
        if pname not in r or pname not in s or pname not in d:
            raise RuntimeError(f"Missing data for product '{pname}'.")
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    for i in I:
        m.addConstr(x_vars[i] <= s[i], name='')
        m.addConstr(x_vars[i] <= d[i], name='')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for i in I:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(m.Status)
    return m
m = build_and_solve(CSVQA_DATA)