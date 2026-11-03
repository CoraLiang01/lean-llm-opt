CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'format': None,
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
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    I = []
    R = {}
    D = {}
    S = {}
    for rec in records:
        values = rec['values']
        pname = values['Product Name']
        I.append(pname)
        R[pname] = float(values['Revenue'])
        D[pname] = int(values['Demand'])
        S[pname] = int(values['Initial Inventory'])
    for pname in I:
        if pname not in R or pname not in D or pname not in S:
            raise ValueError(f'Missing parameter for {pname}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=1, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((R[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= D[i] for i in I), name='')
    m.addConstrs((x[i] <= S[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()