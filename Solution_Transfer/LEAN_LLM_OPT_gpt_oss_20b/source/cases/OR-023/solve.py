CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products with revenue data in the ‘Revenue’ column. The '
          'retailer aims to maximize total revenue using the initial inventory of products classified under ‘ELE-S’. '
          'Inventory levels are given in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘ELE-S’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesStoreoverview.csv',
             'filters': {'conditions': [{'column': 'Product_Reference',
                                         'dtype': 'string',
                                         'evidence': 'ELE-S',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'ELE-ACC-'}],
                         'logic': 'and'},
             'original_rows': 161,
             'records': [{'source_row': 0,
                          'values': {'Demand': '140',
                                     'Initial Inventory': '1000.0',
                                     'Product_Reference': 'ELE-ACC-10000478',
                                     'Revenue': '2.0'}},
                         {'source_row': 1,
                          'values': {'Demand': '136',
                                     'Initial Inventory': '1000.0',
                                     'Product_Reference': 'ELE-ACC-10002424',
                                     'Revenue': '2.0'}},
                         {'source_row': 2,
                          'values': {'Demand': '268',
                                     'Initial Inventory': '2000.0',
                                     'Product_Reference': 'ELE-ACC-10018567',
                                     'Revenue': '2.0'}},
                         {'source_row': 3,
                          'values': {'Demand': '121',
                                     'Initial Inventory': '1000.0',
                                     'Product_Reference': 'ELE-ACC-10019567',
                                     'Revenue': '2.0'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    if not tables:
        raise ValueError('No tables found in CSVQA_DATA')
    table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records in the required table')
    I = []
    r = {}
    d = {}
    s = {}
    required_columns = {'Product_Reference', 'Revenue', 'Demand', 'Initial Inventory'}
    for rec in records:
        values = rec.get('values', {})
        if not required_columns.issubset(values):
            missing = required_columns - values.keys()
            raise ValueError(f'Missing columns in record {rec.get('source_row')}: {missing}')
        prod_ref = values['Product_Reference']
        if prod_ref in I:
            raise ValueError(f'Duplicate product reference found: {prod_ref}')
        I.append(prod_ref)
        try:
            r[prod_ref] = float(values['Revenue'])
            d[prod_ref] = int(values['Demand'])
            s[prod_ref] = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {rec.get('source_row')}: {e}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= d[i] for i in I), name='')
    model.addConstrs((x[i] <= s[i] for i in I), name='')
    model.optimize()
    return model
m = solve_problem()