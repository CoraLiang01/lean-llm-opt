CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records returned for Baby products')
    I = [rec['source_row'] for rec in records]
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        key = rec['source_row']
        vals = rec.get('values', {})
        if 'Revenue' not in vals:
            raise ValueError(f'Missing Revenue for record {key}')
        if 'Demand' not in vals:
            raise ValueError(f'Missing Demand for record {key}')
        if 'Initial Inventory' not in vals:
            raise ValueError(f'Missing Initial Inventory for record {key}')
        try:
            rev[key] = float(vals['Revenue'])
            dem[key] = float(vals['Demand'])
            inv[key] = float(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {key}: {e}')
    m = gurobipy.Model('Baby_Product_Revenue_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= dem[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inv[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()