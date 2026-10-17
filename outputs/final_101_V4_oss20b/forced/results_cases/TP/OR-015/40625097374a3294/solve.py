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
 'route': 'TP',
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
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        vals = rec.get('values', {})
        prod = vals.get('Product Name')
        if prod is None:
            raise ValueError("Record missing 'Product Name'")
        try:
            rev = float(vals.get('Revenue', 0))
            dem = float(vals.get('Demand', 0))
            inv = float(vals.get('Initial Inventory', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f'Invalid numeric value in record for product {prod}') from e
        revenue[prod] = rev
        demand[prod] = dem
        inventory[prod] = inv
    prod_set = set(revenue.keys())
    if not prod_set.issubset(demand.keys()) or not prod_set.issubset(inventory.keys()):
        missing = prod_set - demand.keys() | prod_set - inventory.keys()
        raise ValueError(f'Missing parameters for products: {missing}')
    m = gurobipy.Model('AalopRevenueMaximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(prod_set, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((revenue[p] * x[p] for p in prod_set)), GRB.MAXIMIZE)
    m.addConstrs((x[p] <= demand[p] for p in prod_set), name='')
    m.addConstrs((x[p] <= inventory[p] for p in prod_set), name='')
    m.optimize()
    return m
m = solve_problem()