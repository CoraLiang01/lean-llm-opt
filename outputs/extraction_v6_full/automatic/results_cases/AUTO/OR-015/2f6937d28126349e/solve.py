CSVQA_DATA = {'bindings': [{'index_columns': ['Product Name'],
               'parameter': 'profit',
               'table_id': 'file_0_view_0',
               'value_column': 'Revenue'},
              {'index_columns': ['Product Name'],
               'parameter': 'demand',
               'table_id': 'file_0_view_0',
               'value_column': 'Demand'},
              {'index_columns': ['Product Name'],
               'parameter': 'initial_inventory',
               'table_id': 'file_0_view_0',
               'value_column': 'Initial Inventory'}],
 'ignored_file_indices': [],
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
                                         'evidence': 'products classified under ‘Aalop’',
                                         'format': None,
                                         'inclusive': 'both',
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
             'role': 'products',
             'table_id': 'file_0_view_0'}],
 'validation': {'binding_checks': [{'index_columns': ['Product Name'],
                                    'key_count': 1,
                                    'parameter': 'profit',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Revenue'},
                                   {'index_columns': ['Product Name'],
                                    'key_count': 1,
                                    'parameter': 'demand',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Demand'},
                                   {'index_columns': ['Product Name'],
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
    data = CSVQA_DATA
    table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Required table file_0_view_0 not found in CSVQA_DATA.')
    records = table['records']
    items = []
    revenue = {}
    demand = {}
    initial_inventory = {}
    for rec in records:
        pname = rec['values']['Product Name']
        if pname.startswith('Aalop'):
            items.append(pname)
            try:
                revenue[pname] = float(rec['values']['Revenue'])
                demand[pname] = int(float(rec['values']['Demand']))
                initial_inventory[pname] = int(float(rec['values']['Initial Inventory']))
            except Exception as e:
                raise ValueError(f'Invalid data for product {pname}: {e}')
    for pname in items:
        if pname not in revenue or pname not in demand or pname not in initial_inventory:
            raise ValueError(f'Missing data for product {pname}.')
    m = gp.Model('Aalop_Inventory_Optimization')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='x')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= initial_inventory[i] for i in items), name='inventory')
    m.addConstrs((x[i] <= demand[i] for i in items), name='demand')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()