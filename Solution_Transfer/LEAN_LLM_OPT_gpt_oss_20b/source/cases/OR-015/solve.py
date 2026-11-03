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
    tables = CSVQA_DATA['tables']
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA.")
    records = target_table.get('records', [])
    revenue_dict = {}
    demand_dict = {}
    inventory_dict = {}
    product_list = []
    for rec in records:
        values = rec.get('values', {})
        product_name = values.get('Product Name')
        if product_name is None:
            continue
        if not product_name.startswith('Aalop'):
            continue
        product_list.append(product_name)
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if rev_str is None or dem_str is None or inv_str is None:
            raise ValueError(f'Missing parameter for product {product_name}.')
        try:
            revenue_dict[product_name] = int(rev_str)
            demand_dict[product_name] = int(dem_str)
            inventory_dict[product_name] = float(inv_str)
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for product {product_name}: {e}')
    prod_set = set(product_list)
    if not prod_set.issubset(revenue_dict.keys()):
        missing = prod_set - revenue_dict.keys()
        raise ValueError(f'Missing revenue for products: {missing}')
    if not prod_set.issubset(demand_dict.keys()):
        missing = prod_set - demand_dict.keys()
        raise ValueError(f'Missing demand for products: {missing}')
    if not prod_set.issubset(inventory_dict.keys()):
        missing = prod_set - inventory_dict.keys()
        raise ValueError(f'Missing inventory for products: {missing}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_list, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in product_list)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= demand_dict[i] for i in product_list), name='')
    model.addConstrs((x[i] <= inventory_dict[i] for i in product_list), name='')
    model.optimize()
    return model
m = solve_problem()