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
 'tables': [{'columns': ['Revenue', 'Demand', 'Initial Inventory', 'Product Name'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == table_id), None)
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    records = table.get('records', [])
    product_names = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product Name')
        if prod_name is None:
            raise ValueError('Missing Product Name in record')
        if prod_name in product_names:
            raise ValueError(f'Duplicate product name detected: {prod_name}')
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if rev_str is None or dem_str is None or inv_str is None:
            raise ValueError(f'Missing parameter for product {prod_name}')
        try:
            rev_val = float(rev_str)
            dem_val = int(dem_str)
            inv_val = int(inv_str)
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for product {prod_name}: {e}')
        product_names.append(prod_name)
        revenue[prod_name] = rev_val
        demand[prod_name] = dem_val
        inventory[prod_name] = inv_val
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=1, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in product_names), name='')
    m.addConstrs((x[i] <= inventory[i] for i in product_names), name='')
    m.optimize()
    return m
m = solve_problem()