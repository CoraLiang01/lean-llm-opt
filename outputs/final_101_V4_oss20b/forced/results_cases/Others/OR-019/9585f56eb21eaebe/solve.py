CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'inclusive': 'both',
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    I = [rec['source_row'] for rec in records]
    r = {}
    d = {}
    s = {}
    for rec in records:
        i = rec['source_row']
        try:
            r[i] = float(rec['values']['Revenue'])
            d[i] = int(rec['values']['Demand'])
            s[i] = int(rec['values']['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing parameter for record {i}: {e}')
    for i in I:
        if i not in r or i not in d or i not in s:
            raise ValueError(f'Missing data for product {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.addConstrs((x[i] <= s[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()