CSVQA_DATA = {'bindings': [{'index_columns': ['Product Name'],
               'parameter': 'revenue',
               'table_id': 'file_0_view_0',
               'value_column': 'Revenue'},
              {'index_columns': ['Product Name'],
               'parameter': 'initial_inventory',
               'table_id': 'file_0_view_0',
               'value_column': 'Initial Inventory'},
              {'index_columns': ['Product Name'],
               'parameter': 'demand',
               'table_id': 'file_0_view_0',
               'value_column': 'Demand'}],
 'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘27in’',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '261.2933'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965'}}],
             'returned_rows': 2,
             'role': 'products',
             'table_id': 'file_0_view_0'}],
 'validation': {'binding_checks': [{'index_columns': ['Product Name'],
                                    'key_count': 2,
                                    'parameter': 'revenue',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Revenue'},
                                   {'index_columns': ['Product Name'],
                                    'key_count': 2,
                                    'parameter': 'initial_inventory',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Initial Inventory'},
                                   {'index_columns': ['Product Name'],
                                    'key_count': 2,
                                    'parameter': 'demand',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Demand'}],
                'matrix_checks': [],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    items = []
    revenue = {}
    inventory = {}
    demand = {}
    for rec in records:
        pname = rec['values']['Product Name']
        if pname.startswith('27in'):
            items.append(pname)
            try:
                revenue[pname] = float(rec['values']['Revenue'])
                inventory[pname] = int(rec['values']['Initial Inventory'])
                demand[pname] = int(rec['values']['Demand'])
            except Exception as e:
                raise ValueError(f'Invalid data for product {pname}: {e}')
    for pname in items:
        if pname not in revenue or pname not in inventory or pname not in demand:
            raise ValueError(f'Missing data for product {pname}')
    m = gp.Model('27in_Promo')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='x')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='inventory')
    m.addConstrs((x[i] <= demand[i] for i in items), name='demand')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()