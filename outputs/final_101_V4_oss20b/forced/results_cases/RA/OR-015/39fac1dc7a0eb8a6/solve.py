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
 'route': 'RA',
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
    index_set = []
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product Name')
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if prod_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec}")
        if rev_str is None:
            raise ValueError(f"Missing 'Revenue' for product {prod_name}")
        if dem_str is None:
            raise ValueError(f"Missing 'Demand' for product {prod_name}")
        if inv_str is None:
            raise ValueError(f"Missing 'Initial Inventory' for product {prod_name}")
        try:
            rev_val = float(rev_str)
        except ValueError:
            raise ValueError(f"Invalid numeric value for Revenue '{rev_str}' in product {prod_name}")
        try:
            dem_val = float(dem_str)
        except ValueError:
            raise ValueError(f"Invalid numeric value for Demand '{dem_str}' in product {prod_name}")
        try:
            inv_val = float(inv_str)
        except ValueError:
            raise ValueError(f"Invalid numeric value for Initial Inventory '{inv_str}' in product {prod_name}")
        if prod_name in revenue:
            raise ValueError(f'Duplicate product name detected: {prod_name}')
        revenue[prod_name] = rev_val
        demand[prod_name] = dem_val
        inventory[prod_name] = inv_val
        index_set.append(prod_name)
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(index_set, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in index_set)), GRB.MAXIMIZE)
    for i in index_set:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')