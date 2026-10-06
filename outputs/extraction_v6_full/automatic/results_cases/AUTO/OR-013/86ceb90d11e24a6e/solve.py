CSVQA_DATA = {'bindings': [{'index_columns': ['Product Name'],
               'parameter': 'revenue',
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
 'query': 'This supermarket offers a variety of best-selling products, and the specific revenue information is shown '
          'in the table, with relevant data provided in the “Revenue” column. Each product has its own demand level. '
          'The retailer’s objective is to maximize the total revenue by focusing on the sales volume of “4U” products. '
          'The initial inventory levels of these products are detailed in the “Initial Inventory” column. During the '
          'sales period, restocking is not allowed and there is no in-transit inventory.\n'
          '\n'
          'Demand for the “4U” products during the sales horizon is assumed to be deterministic and known in advance, '
          'with demand quantities specified in the “Demand” column. The decision variable x_i represents the number of '
          'units of each “4U” product i that the retailer plans to fulfill, where each x_i is a non-negative integer. '
          'Because the fulfillment quantity cannot exceed either the available inventory or the realized demand, the '
          'decision variables must satisfy both inventory and demand constraints.\n'
          '\n'
          'The retailer therefore aims to determine the optimal fulfillment quantities in order to maximize total '
          'revenue while respecting both inventory availability and demand limits.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineSalesinUSA.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'sales volume of “4U” products',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': '4U'}],
                         'logic': 'and'},
             'original_rows': 47932,
             'records': [{'source_row': 1,
                          'values': {'Demand': '5',
                                     'Initial Inventory': '30',
                                     'Product Name': '4U_Service_22',
                                     'Revenue': '56.0'}},
                         {'source_row': 2,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': '4U_Service_36',
                                     'Revenue': '21.6'}},
                         {'source_row': 3,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': '4U_Service_7',
                                     'Revenue': '62.5'}}],
             'returned_rows': 3,
             'role': 'product_data',
             'table_id': 'file_0_view_0'}],
 'validation': {'binding_checks': [{'index_columns': ['Product Name'],
                                    'key_count': 3,
                                    'parameter': 'revenue',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Revenue'},
                                   {'index_columns': ['Product Name'],
                                    'key_count': 3,
                                    'parameter': 'demand',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Demand'},
                                   {'index_columns': ['Product Name'],
                                    'key_count': 3,
                                    'parameter': 'initial_inventory',
                                    'status': 'OK',
                                    'table_id': 'file_0_view_0',
                                    'value_column': 'Initial Inventory'}],
                'matrix_checks': [],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
    records = []
    for rec in table['records']:
        pname = rec['values']['Product Name']
        if pname.startswith('4U'):
            records.append(rec)
    items = []
    revenue = {}
    demand = {}
    initial_inventory = {}
    for rec in records:
        pname = rec['values']['Product Name']
        items.append(pname)
        try:
            revenue[pname] = float(rec['values']['Revenue'])
            demand[pname] = int(rec['values']['Demand'])
            initial_inventory[pname] = int(rec['values']['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {pname}: {e}')
    for pname in items:
        if pname not in revenue or pname not in demand or pname not in initial_inventory:
            raise ValueError(f'Missing data for product {pname}')
    m = gp.Model('4U_Product_Fulfillment')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='x')
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