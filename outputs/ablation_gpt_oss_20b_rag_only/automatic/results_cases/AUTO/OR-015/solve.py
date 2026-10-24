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
                                         'evidence': 'Aalop',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table = None
    for t in CSVQA_DATA['tables']:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Required table file_0_view_0 not found')
    records = table.get('records', [])
    I = []
    r = {}
    d = {}
    inv = {}
    for rec in records:
        values = rec['values']
        prod = values['Product Name']
        I.append(prod)
        r[prod] = float(values['Revenue'])
        d[prod] = int(values['Demand'])
        inv[prod] = float(values['Initial Inventory'])
    for prod in I:
        if prod not in r or prod not in d or prod not in inv:
            raise ValueError(f'Missing parameter for {prod}')
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    for prod in I:
        m.addConstr(x[prod] <= d[prod])
        m.addConstr(x[prod] <= inv[prod])
    m.setObjective(gurobipy.quicksum((r[prod] * x[prod] for prod in I)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()