CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'DifferentStoreSales.csv',
             'filters': {'conditions': [{'column': 'Product_Name',
                                         'dtype': 'string',
                                         'evidence': 'Books',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Books'}],
                         'logic': 'and'},
             'original_rows': 40,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1980',
                                     'Initial Inventory': '9920.0',
                                     'Product_Name': 'Books_15.15',
                                     'Revenue': '15.15'}},
                         {'source_row': 1,
                          'values': {'Demand': '3024',
                                     'Initial Inventory': '20160.0',
                                     'Product_Name': 'Books_30.3',
                                     'Revenue': '30.3'}},
                         {'source_row': 2,
                          'values': {'Demand': '4536',
                                     'Initial Inventory': '30000.0',
                                     'Product_Name': 'Books_45.45',
                                     'Revenue': '45.45'}},
                         {'source_row': 3,
                          'values': {'Demand': '5601',
                                     'Initial Inventory': '38360.0',
                                     'Product_Name': 'Books_60.6',
                                     'Revenue': '60.6'}},
                         {'source_row': 4,
                          'values': {'Demand': '7567',
                                     'Initial Inventory': '51450.0',
                                     'Product_Name': 'Books_75.75',
                                     'Revenue': '75.75'}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == table_id:
            table = tbl
            break
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    ids = []
    revenue = {}
    demand = {}
    inventory = {}
    for record in table['records']:
        prod_name = record['values']['Product_Name']
        ids.append(prod_name)
        try:
            revenue[prod_name] = float(record['values']['Revenue'])
            demand[prod_name] = float(record['values']['Demand'])
            inventory[prod_name] = float(record['values']['Initial Inventory'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid data for product {prod_name}') from e
    for pid in ids:
        if pid not in revenue:
            raise ValueError(f'Missing revenue for product {pid}')
        if pid not in demand:
            raise ValueError(f'Missing demand for product {pid}')
        if pid not in inventory:
            raise ValueError(f'Missing inventory for product {pid}')
    m = gurobipy.Model('BooksRevenueMax')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    for i in ids:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()