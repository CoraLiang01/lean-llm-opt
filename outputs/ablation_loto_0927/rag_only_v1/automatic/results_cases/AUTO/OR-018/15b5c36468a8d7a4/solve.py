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
                                         'operator': 'prefix',
                                         'value': 'Baby'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '"Baby"',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'or'},
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
from gurobipy import Model, GRB
import re

def solve_baby_product_optimization(CSVQA_DATA):
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
        pname = rec['values']['Product Name']
        if re.search('\\bBaby\\b', pname) or pname.startswith('Baby'):
            idx = pname
            I.append(idx)
            try:
                r[idx] = float(rec['values']['Revenue'])
                d[idx] = int(rec['values']['Demand'])
                s[idx] = int(rec['values']['Initial Inventory'])
            except Exception as e:
                raise RuntimeError(f"Error parsing parameters for product '{idx}': {e}")
    for idx in I:
        if idx not in r or idx not in d or idx not in s:
            raise RuntimeError(f"Missing parameter for product '{idx}' in index set I.")
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= s[i] for i in I), name='')
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_baby_product_optimization(CSVQA_DATA)