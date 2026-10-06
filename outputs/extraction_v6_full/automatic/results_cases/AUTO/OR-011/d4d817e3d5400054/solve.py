CSVQA_DATA = {'bindings': [{'index_columns': ['id_number'],
               'parameter': 'revenue',
               'table_id': 'file_0_view_0',
               'value_column': 'Revenue'},
              {'index_columns': ['id_number'],
               'parameter': 'demand',
               'table_id': 'file_0_view_0',
               'value_column': 'Demand'},
              {'index_columns': ['id_number'],
               'parameter': 'initial_inventory',
               'table_id': 'file_0_view_0',
               'value_column': 'Initial Inventory'}],
 'ignored_file_indices': [],
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
 'route': 'NRM',
 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘id999’',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'exact',
                                         'value': 'id999'}],
                         'logic': 'and'},
             'original_rows': 900,
             'records': [{'source_row': 899,
                          'values': {'Demand': '8171',
                                     'Initial Inventory': '56450',
                                     'Revenue': '434.74',
                                     'id_number': 'id999'}}],
             'returned_rows': 1,
             'role': 'products',
             'table_id': 'file_0_view_0'}],
 'validation': {'binding_checks': [{'index_columns': ['id_number'],
                                    'key_count': 1,
                                    'parameter': 'revenue',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Revenue'},
                                   {'index_columns': ['id_number'],
                                    'key_count': 1,
                                    'parameter': 'demand',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Demand'},
                                   {'index_columns': ['id_number'],
                                    'key_count': 1,
                                    'parameter': 'initial_inventory',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Initial Inventory'}],
                'matrix_checks': [],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data_table = None
    for table in CSVQA_DATA['tables']:
        if table['table_id'] == 'file_0_view_0':
            data_table = table
            break
    if data_table is None:
        raise ValueError('Required table file_0_view_0 not found in CSVQA_DATA.')
    records = [rec for rec in data_table['records'] if rec['values']['id_number'] == 'id999']
    if not records:
        raise ValueError("No records found for id_number == 'id999'.")
    I = []
    r = {}
    d = {}
    s = {}
    for rec in records:
        i = rec['values']['id_number']
        I.append(i)
        try:
            r[i] = float(rec['values']['Revenue'])
            d[i] = int(rec['values']['Demand'])
            s[i] = int(rec['values']['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Error parsing parameters for {i}: {e}')
    for i in I:
        if i not in r or i not in d or i not in s:
            raise ValueError(f'Missing parameter(s) for product {i}.')
    m = gp.Model('Supermarket_id999_Allocation')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='x')
    m.setObjective(gp.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= s[i] for i in I), name='inventory')
    m.addConstrs((x[i] <= d[i] for i in I), name='demand')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()