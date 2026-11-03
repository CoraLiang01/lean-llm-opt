CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Revenue', 'Demand', 'Initial Inventory', 'Product Name'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'classified under ‘27in’',
                                         'operator': 'contains',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be fulfilled.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['Revenue', 'Demand', 'Initial Inventory', 'Product Name'], 'file_index': 0, 'file_name': 'Salesorders.csv', 'filters': {'conditions': [{'column': 'Product Name', 'dtype': 'string', 'evidence': 'classified under ‘27in’', 'operator': 'contains', 'value': '27in'}], 'logic': 'and'}, 'original_rows': 19, 'records': [{'source_row': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933'}}, {'source_row': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '52.4965'}}], 'returned_rows': 2, 'role': 'descriptive non-unique role', 'table_id': 'file_0_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA.")
    revenue = {}
    demand = {}
    inventory = {}
    product_names = []
    for record in table['records']:
        values = record['values']
        prod = values['Product Name']
        product_names.append(prod)
        try:
            rev = float(values['Revenue'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Revenue for product {prod}')
        try:
            dem = int(values['Demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Demand for product {prod}')
        try:
            inv = int(values['Initial Inventory'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Initial Inventory for product {prod}')
        revenue[prod] = rev
        demand[prod] = dem
        inventory[prod] = inv
    keys_set = set(product_names)
    if set(revenue.keys()) != keys_set:
        raise ValueError('Revenue keys mismatch.')
    if set(demand.keys()) != keys_set:
        raise ValueError('Demand keys mismatch.')
    if set(inventory.keys()) != keys_set:
        raise ValueError('Inventory keys mismatch.')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(keys_set, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in keys_set)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in keys_set), name='')
    m.addConstrs((x[i] <= inventory[i] for i in keys_set), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()