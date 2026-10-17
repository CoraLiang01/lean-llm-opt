CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'An e-commerce platform sells women’s clothing items with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘FAUX’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent '
          'the number of units of each ‘FAUX’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'ZARASales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'FAUX',
                                         'operator': 'contains',
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
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    if not tables:
        raise ValueError('No tables found in CSVQA_DATA')
    table = tables[0]
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the specified table')
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        key = rec['source_row']
        vals = rec['values']
        if 'Revenue' not in vals or 'Demand' not in vals or 'Initial Inventory' not in vals:
            raise ValueError(f'Missing required field in record {key}')
        try:
            r = float(vals['Revenue'])
            d = int(vals['Demand'])
            s = int(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {key}: {e}')
        I.append(key)
        revenue[key] = r
        demand[key] = d
        inventory[key] = s
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()