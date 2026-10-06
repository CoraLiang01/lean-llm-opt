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
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise RuntimeError('Required table_id file_0_view_0 not found in CSVQA_DATA.')
    records = table['records']
    I = []
    for rec in records:
        pname = rec['values']['Product Name']
        if not pname.casefold().startswith('baby'):
            continue
        I.append(pname)
    r = {}
    d = {}
    s = {}
    for rec in records:
        pname = rec['values']['Product Name']
        if pname not in I:
            continue
        try:
            r[pname] = float(rec['values']['Revenue'])
            d[pname] = float(rec['values']['Demand'])
            s[pname] = float(rec['values']['Initial Inventory'])
        except Exception as e:
            raise RuntimeError(f'Missing or invalid data for product {pname}: {e}')
    for i in I:
        if i not in r or i not in d or i not in s:
            raise RuntimeError(f'Missing parameter for product {i}')
    ub = {}
    for i in I:
        ub[i] = min(d[i], s[i])
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, lb=0, ub=ub, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for i in I:
            print(x[i].VarName, x[i].X)
    else:
        print(m.Status)
    return m
m = solve_problem(CSVQA_DATA)