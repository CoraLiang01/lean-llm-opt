CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Baby’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory for Baby products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB
import re

def solve_baby_product_optimization(CSVQA_DATA):
    table_id = 'file_0_view_0'
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == table_id:
            table = t
            break
    if table is None:
        raise RuntimeError('Required table_id not found in CSVQA_DATA.')
    records = table['records']
    columns = table['columns']
    I = []
    r = {}
    d = {}
    s = {}
    prefix = 'Baby'
    for rec in records:
        vals = rec['values']
        pname = vals['Product Name']
        if not re.match('^Baby', pname):
            continue
        I.append(pname)
        try:
            r[pname] = float(vals['Revenue'])
            d[pname] = float(vals['Demand'])
            s[pname] = float(vals['Initial Inventory'])
        except Exception as e:
            raise RuntimeError(f'Missing or invalid parameter for product {pname}: {e}')
    for pname in I:
        if pname not in r or pname not in d or pname not in s:
            raise RuntimeError(f'Missing data for product {pname} in index set I.')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(sum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= s[i] for i in I), name='')
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName}: {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_baby_product_optimization(CSVQA_DATA)