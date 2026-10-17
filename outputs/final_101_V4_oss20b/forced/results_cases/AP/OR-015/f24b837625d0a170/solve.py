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
 'route': 'AP',
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
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'")
    R = {}
    D = {}
    I = {}
    products = []
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product Name')
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if prod_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec}")
        if rev_str is None:
            raise ValueError(f"Missing 'Revenue' for product '{prod_name}'")
        if dem_str is None:
            raise ValueError(f"Missing 'Demand' for product '{prod_name}'")
        if inv_str is None:
            raise ValueError(f"Missing 'Initial Inventory' for product '{prod_name}'")
        try:
            rev = float(rev_str)
            dem = float(dem_str)
            inv = float(inv_str)
        except ValueError as e:
            raise ValueError(f"Numeric conversion error for product '{prod_name}': {e}")
        if prod_name in products:
            raise ValueError(f"Duplicate product name '{prod_name}' detected")
        products.append(prod_name)
        R[prod_name] = rev
        D[prod_name] = dem
        I[prod_name] = inv
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((R[i] * x[i] for i in products)), GRB.MAXIMIZE)
    for i in products:
        m.addConstr(x[i] <= D[i], name=f'demand_{i}')
        m.addConstr(x[i] <= I[i], name=f'inventory_{i}')
        m.addConstr(x[i] >= 0, name=f'nonneg_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()