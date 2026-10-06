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
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Baby’',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'products',
             'table_id': 'file_0_view_0'}],
 'validation': {'binding_checks': [{'index_columns': ['Product Name'],
                                    'key_count': 1,
                                    'parameter': 'revenue',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Revenue'},
                                   {'index_columns': ['Product Name'],
                                    'key_count': 1,
                                    'parameter': 'initial_inventory',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Initial Inventory'},
                                   {'index_columns': ['Product Name'],
                                    'key_count': 1,
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
    initial_inventory = {}
    demand = {}
    for rec in records:
        pname = rec['values']['Product Name']
        if pname.startswith('Baby'):
            items.append(pname)
            try:
                revenue[pname] = float(rec['values']['Revenue'])
                initial_inventory[pname] = int(rec['values']['Initial Inventory'])
                demand[pname] = int(rec['values']['Demand'])
            except Exception as e:
                raise ValueError(f'Invalid data for product {pname}: {e}')
    for pname in items:
        if pname not in revenue or pname not in initial_inventory or pname not in demand:
            raise ValueError(f'Missing data for product {pname}')
    m = gp.Model('Baby_Product_Revenue_Maximization')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='x')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= initial_inventory[i] for i in items), name='inventory')
    m.addConstrs((x[i] <= demand[i] for i in items), name='demand')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()