CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store offers a variety of best-selling products with profit data provided in the ‘Revenue’ column. '
          'The retailer aims to maximize total revenue using the initial inventory of products classified under '
          '‘S700_’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified '
          'in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i '
          'indicate the number of units of each ‘S700_’ product i that the store intends to fulfill.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SampleSalesData.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'S700_',
                                         'operator': 'prefix',
                                         'value': 'S700_'}],
                         'logic': 'and'},
             'original_rows': 109,
             'records': [{'source_row': 95,
                          'values': {'Demand': '1219',
                                     'Initial Inventory': '9020',
                                     'Product Name': 'S700_1138',
                                     'Revenue': '70.67'}},
                         {'source_row': 96,
                          'values': {'Demand': '1127',
                                     'Initial Inventory': '8370',
                                     'Product Name': 'S700_1691',
                                     'Revenue': '100.0'}},
                         {'source_row': 97,
                          'values': {'Demand': '1129',
                                     'Initial Inventory': '8390',
                                     'Product Name': 'S700_1938',
                                     'Revenue': '70.15'}},
                         {'source_row': 98,
                          'values': {'Demand': '1176',
                                     'Initial Inventory': '8680',
                                     'Product Name': 'S700_2047',
                                     'Revenue': '100.0'}},
                         {'source_row': 99,
                          'values': {'Demand': '1301',
                                     'Initial Inventory': '9400',
                                     'Product Name': 'S700_2466',
                                     'Revenue': '100.0'}},
                         {'source_row': 100,
                          'values': {'Demand': '1340',
                                     'Initial Inventory': '9900',
                                     'Product Name': 'S700_2610',
                                     'Revenue': '65.77'}},
                         {'source_row': 101,
                          'values': {'Demand': '1357',
                                     'Initial Inventory': '9760',
                                     'Product Name': 'S700_2824',
                                     'Revenue': '100.0'}},
                         {'source_row': 102,
                          'values': {'Demand': '1158',
                                     'Initial Inventory': '8610',
                                     'Product Name': 'S700_2834',
                                     'Revenue': '100.0'}},
                         {'source_row': 103,
                          'values': {'Demand': '1287',
                                     'Initial Inventory': '9380',
                                     'Product Name': 'S700_3167',
                                     'Revenue': '74.4'}},
                         {'source_row': 104,
                          'values': {'Demand': '1281',
                                     'Initial Inventory': '9170',
                                     'Product Name': 'S700_3505',
                                     'Revenue': '81.14'}},
                         {'source_row': 105,
                          'values': {'Demand': '1135',
                                     'Initial Inventory': '8520',
                                     'Product Name': 'S700_3962',
                                     'Revenue': '100.0'}},
                         {'source_row': 106,
                          'values': {'Demand': '1392',
                                     'Initial Inventory': '10290',
                                     'Product Name': 'S700_4002',
                                     'Revenue': '61.44'}}],
             'returned_rows': 12,
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
        raise ValueError('Table file_0_view_0 not found in CSVQA_DATA')
    records = table.get('records', [])
    product_names = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec['values']
        pname = values.get('Product Name')
        if pname is None:
            raise ValueError('Missing Product Name in a record')
        if pname in product_names:
            raise ValueError(f'Duplicate product name detected: {pname}')
        product_names.append(pname)
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if rev_str is None or dem_str is None or inv_str is None:
            raise ValueError(f'Missing parameter for product {pname}')
        try:
            rev = float(rev_str)
            dem = int(dem_str)
            inv = int(inv_str)
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for product {pname}: {e}')
        revenue[pname] = rev
        demand[pname] = dem
        inventory[pname] = inv
    x = m.addVars(product_names, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    for i in product_names:
        ub = min(demand[i], inventory[i])
        m.addConstr(x[i] <= ub, name=f'ub_{i}')
    m.optimize()
    return m
m = solve_problem()