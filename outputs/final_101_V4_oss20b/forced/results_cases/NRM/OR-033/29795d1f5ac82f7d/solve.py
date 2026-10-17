CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
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
    tables = CSVQA_DATA.get('tables', [])
    table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Table file_0_view_0 not found in CSVQA_DATA')
    records = table.get('records', [])
    if not records:
        raise ValueError('No records returned for table file_0_view_0')
    I = []
    Revenue_i = {}
    Demand_i = {}
    InitialInventory_i = {}
    for rec in records:
        values = rec.get('values', {})
        for col in ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']:
            if col not in values:
                raise ValueError(f'Missing column {col} in record {rec.get('source_row')}')
        prod = values['Product Name']
        if prod in I:
            raise ValueError(f'Duplicate product name {prod} in data')
        I.append(prod)
        try:
            rev = float(values['Revenue'])
        except Exception:
            raise ValueError(f'Invalid Revenue value for product {prod}: {values['Revenue']}')
        try:
            dem = int(values['Demand'])
        except Exception:
            raise ValueError(f'Invalid Demand value for product {prod}: {values['Demand']}')
        try:
            inv = int(values['Initial Inventory'])
        except Exception:
            raise ValueError(f'Invalid Initial Inventory value for product {prod}: {values['Initial Inventory']}')
        Revenue_i[prod] = rev
        Demand_i[prod] = dem
        InitialInventory_i[prod] = inv
    m = gurobipy.Model('BabyProductRevenueMaximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=1, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((Revenue_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= Demand_i[i], name=f'demand_{i}')
        m.addConstr(x[i] <= InitialInventory_i[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()