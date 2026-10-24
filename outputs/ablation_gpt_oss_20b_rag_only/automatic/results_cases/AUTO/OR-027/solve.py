CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'filters': {'conditions': [{'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': 'Organ',
                                         'operator': 'contains',
                                         'value': 'Organ'}],
                         'logic': 'and'},
             'original_rows': 23,
             'records': [{'source_row': 17,
                          'values': {'Demand': '678906',
                                     'Initial Inventory': '5034020.0',
                                     'Revenue': '60.8',
                                     'Sub Category': 'Organic Fruits'}},
                         {'source_row': 18,
                          'values': {'Demand': '749927',
                                     'Initial Inventory': '5589290.0',
                                     'Revenue': '918.45',
                                     'Sub Category': 'Organic Staples'}},
                         {'source_row': 19,
                          'values': {'Demand': '699808',
                                     'Initial Inventory': '5202710.0',
                                     'Revenue': '77.52',
                                     'Sub Category': 'Organic Vegetables'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    data = CSVQA_DATA
    table = None
    for t in data['tables']:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Table file_0_view_0 not found')
    records = table['records']
    I = []
    revenue = {}
    demand = {}
    init_inv = {}
    for rec in records:
        vals = rec['values']
        for col in ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory']:
            if col not in vals:
                raise ValueError(f'Missing column {col} in record {rec}')
        sub = vals['Sub Category']
        I.append(sub)
        revenue[sub] = float(vals['Revenue'])
        demand[sub] = int(vals['Demand'])
        init_inv[sub] = float(vals['Initial Inventory'])
    for i in I:
        if i not in revenue or i not in demand or i not in init_inv:
            raise ValueError(f'Missing parameter for {i}')
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    for i in I:
        m.addConstr(x[i] <= demand[i])
        m.addConstr(x[i] <= init_inv[i])
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.Status)