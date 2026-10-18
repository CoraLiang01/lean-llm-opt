CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products with revenue data in the ‘Revenue’ column. The '
          'retailer aims to maximize total revenue using the initial inventory of products classified under ‘ELE-S’. '
          'Inventory levels are given in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘ELE-S’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesStoreoverview.csv',
             'filters': {'conditions': [{'column': 'Product_Reference',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘ELE-S’',
                                         'operator': 'prefix',
                                         'value': 'ELE-S'}],
                         'logic': 'and'},
             'original_rows': 161,
             'records': [{'source_row': 36,
                          'values': {'Demand': '295',
                                     'Initial Inventory': '2000.0',
                                     'Product_Reference': 'ELE-SMA-10000463',
                                     'Revenue': '4.0'}},
                         {'source_row': 37,
                          'values': {'Demand': '1002',
                                     'Initial Inventory': '7000.0',
                                     'Product_Reference': 'ELE-SMA-10000487',
                                     'Revenue': '14.0'}},
                         {'source_row': 38,
                          'values': {'Demand': '958',
                                     'Initial Inventory': '7000.0',
                                     'Product_Reference': 'ELE-SMA-10003333',
                                     'Revenue': '14.0'}},
                         {'source_row': 39,
                          'values': {'Demand': '777',
                                     'Initial Inventory': '6000.0',
                                     'Product_Reference': 'ELE-SMA-10009012',
                                     'Revenue': '4.0'}},
                         {'source_row': 40,
                          'values': {'Demand': '271',
                                     'Initial Inventory': '2000.0',
                                     'Product_Reference': 'ELE-SMA-10009999',
                                     'Revenue': '4.0'}},
                         {'source_row': 41,
                          'values': {'Demand': '244',
                                     'Initial Inventory': '2000.0',
                                     'Product_Reference': 'ELE-SMA-10011234',
                                     'Revenue': '4.0'}},
                         {'source_row': 42,
                          'values': {'Demand': '990',
                                     'Initial Inventory': '7000.0',
                                     'Product_Reference': 'ELE-SMA-10027456',
                                     'Revenue': '14.0'}},
                         {'source_row': 43,
                          'values': {'Demand': '1000',
                                     'Initial Inventory': '7000.0',
                                     'Product_Reference': 'ELE-SMA-10028567',
                                     'Revenue': '14.0'}},
                         {'source_row': 44,
                          'values': {'Demand': '169',
                                     'Initial Inventory': '1200.0',
                                     'Product_Reference': 'ELE-SPE-10000484',
                                     'Revenue': '2.4'}},
                         {'source_row': 45,
                          'values': {'Demand': '155',
                                     'Initial Inventory': '1200.0',
                                     'Product_Reference': 'ELE-SPE-10003030',
                                     'Revenue': '2.4'}},
                         {'source_row': 46,
                          'values': {'Demand': '327',
                                     'Initial Inventory': '2400.0',
                                     'Product_Reference': 'ELE-SPE-10024123',
                                     'Revenue': '2.4'}},
                         {'source_row': 47,
                          'values': {'Demand': '174',
                                     'Initial Inventory': '1200.0',
                                     'Product_Reference': 'ELE-SPE-10025234',
                                     'Revenue': '2.4'}}],
             'returned_rows': 12,
             'role': 'descriptive non-unique role',
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
import re
records = []
for rec in table['records']:
    ref = rec['values']['Product_Reference']
    if re.match('^ELE-S', ref):
        records.append(rec)
I = []
r = {}
s = {}
d = {}
for rec in records:
    values = rec['values']
    i = values['Product_Reference']
    try:
        r_i = float(values['Revenue'])
        s_i = float(values['Initial Inventory'])
        d_i = float(values['Demand'])
    except Exception as e:
        raise RuntimeError(f'Invalid numeric data for product {i}: {e}')
    I.append(i)
    r[i] = r_i
    s[i] = s_i
    d[i] = d_i
if not I:
    raise RuntimeError("No products found with Product_Reference prefix 'ELE-S'.")
for i in I:
    if i not in r or i not in s or i not in d:
        raise RuntimeError(f'Missing parameter for product {i}.')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub=None, name='')
    for i in I:
        m.addConstr(x_vars[i] <= s[i], name='')
        m.addConstr(x_vars[i] <= d[i], name='')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in I:
            v = x_vars[i]
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_model()