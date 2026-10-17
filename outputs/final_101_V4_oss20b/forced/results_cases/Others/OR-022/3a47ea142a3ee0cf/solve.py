CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
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

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    I = [rec['source_row'] for rec in records]
    r = {}
    d = {}
    s = {}
    for rec in records:
        i = rec['source_row']
        vals = rec['values']
        try:
            r[i] = float(vals['Revenue'])
            d[i] = int(vals['Demand'])
            s[i] = int(vals['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing parameter {e} for record {i}')
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i], name=f'max_demand_{i}')
        m.addConstr(x[i] <= s[i], name=f'max_inventory_{i}')
    m.optimize()
    return m
m = solve_problem()