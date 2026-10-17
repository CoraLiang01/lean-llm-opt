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
 'route': 'RA',
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
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f'Table with id {table_id} not found in CSVQA_DATA')
    records = target_table.get('records', [])
    I = []
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        values = rec.get('values', {})
        id_num = values.get('id_number', '').strip()
        if not id_num.startswith('id999'):
            continue
        I.append(id_num)
        try:
            rev_val = float(values.get('Revenue', '0'))
            dem_val = int(values.get('Demand', '0'))
            inv_val = int(values.get('Initial Inventory', '0'))
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid numeric value in record {rec['source_row']}: {e}')
        rev[id_num] = rev_val
        dem[id_num] = dem_val
        inv[id_num] = inv_val
    if not I:
        raise ValueError("No records with id_number prefix 'id999' found")
    for i in I:
        if i not in rev:
            raise ValueError(f'Missing revenue for product {i}')
        if i not in dem:
            raise ValueError(f'Missing demand for product {i}')
        if i not in inv:
            raise ValueError(f'Missing inventory for product {i}')
    m = gurobipy.Model('MaxRevenue')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= inv[i], name=f'inv_{i}')
        m.addConstr(x[i] <= dem[i], name=f'dem_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')