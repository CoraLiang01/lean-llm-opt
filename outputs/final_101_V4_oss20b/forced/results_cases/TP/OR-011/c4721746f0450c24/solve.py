CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the '
          '‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective '
          'is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. '
          'The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. '
          'During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n'
          '\n'
          'Demand for each product during the sales period is assumed to be deterministic and known in advance, with '
          'demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of '
          'units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative '
          'integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the '
          'fulfillment quantities must satisfy both inventory and demand constraints.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': 'id999',
                                         'operator': 'prefix',
                                         'value': 'id999'}],
                         'logic': 'and'},
             'original_rows': 900,
             'records': [{'source_row': 899,
                          'values': {'Demand': '8171',
                                     'Initial Inventory': '56450',
                                     'Revenue': '434.74',
                                     'id_number': 'id999'}}],
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
    I = set()
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec.get('values', {})
        id_number = values.get('id_number', '')
        if not isinstance(id_number, str):
            continue
        if not id_number.startswith('id999'):
            continue
        i = id_number
        I.add(i)
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if rev_str is None or dem_str is None or inv_str is None:
            raise ValueError(f'Missing parameter for product {i}')
        try:
            rev_val = float(rev_str)
            dem_val = int(dem_str)
            inv_val = int(inv_str)
        except Exception as e:
            raise ValueError(f'Invalid numeric value for product {i}: {e}')
        revenue[i] = rev_val
        demand[i] = dem_val
        inventory[i] = inv_val
    if not I:
        raise ValueError("No products with id starting with 'id999' found")
    for i in I:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing data for product {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= demand[i] for i in I), name='')
    model.addConstrs((x[i] <= inventory[i] for i in I), name='')
    model.optimize()
    return model
m = solve_problem()