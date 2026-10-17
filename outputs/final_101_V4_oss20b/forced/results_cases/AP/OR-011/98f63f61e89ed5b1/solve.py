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
 'route': 'AP',
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
    tables = CSVQA_DATA['tables']
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError('Table file_0_view_0 not found')
    records = target_table.get('records', [])
    I = []
    r_i = {}
    d_i = {}
    v_i = {}
    for rec in records:
        values = rec.get('values', {})
        id_num = values.get('id_number')
        if id_num is None:
            raise ValueError('Missing id_number')
        rev = values.get('Revenue')
        dem = values.get('Demand')
        inv = values.get('Initial Inventory')
        if rev is None or dem is None or inv is None:
            raise ValueError(f'Missing parameter for {id_num}')
        try:
            r_val = float(rev)
            d_val = int(dem)
            v_val = int(inv)
        except ValueError:
            raise ValueError(f'Invalid numeric value for {id_num}')
        I.append(id_num)
        r_i[id_num] = r_val
        d_i[id_num] = d_val
        v_i[id_num] = v_val
    if not I:
        raise ValueError('No records after filtering')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.addConstrs((x[i] <= d_i[i] for i in I), name='')
    m.addConstrs((x[i] <= v_i[i] for i in I), name='')
    m.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()