CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'inclusive': 'both',
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    rev = {}
    dem = {}
    inv = {}
    for record in target_table.get('records', []):
        values = record.get('values', {})
        product_name = values.get('Product Name')
        if product_name is None:
            continue
        try:
            rev_val = float(values.get('Revenue', 0))
            dem_val = int(values.get('Demand', 0))
            inv_val = int(values.get('Initial Inventory', 0))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric data for product '{product_name}'")
        rev[product_name] = rev_val
        dem[product_name] = dem_val
        inv[product_name] = inv_val
    if set(rev.keys()) != set(dem.keys()) or set(rev.keys()) != set(inv.keys()):
        missing = set(rev.keys()).symmetric_difference(set(dem.keys()).union(set(inv.keys())))
        raise ValueError(f'Missing parameter data for products: {missing}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(rev.keys(), lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in rev)), GRB.MAXIMIZE)
    for i in rev:
        model.addConstr(x[i] <= dem[i], name=f'dem_{i}')
        model.addConstr(x[i] <= inv[i], name=f'inv_{i}')
    model.optimize()
    return model
m = solve_problem()