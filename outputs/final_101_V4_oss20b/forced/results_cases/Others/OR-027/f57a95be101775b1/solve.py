CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'Others',
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
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    ids = []
    r_i = {}
    d_i = {}
    s_i = {}
    for rec in records:
        values = rec.get('values', {})
        subcat = values.get('Sub Category', '')
        if 'organ' not in subcat.lower():
            continue
        i = rec.get('source_row')
        if i is None:
            raise ValueError('Record missing source_row identifier')
        ids.append(i)
        try:
            r_val = float(values.get('Revenue', '0').replace(',', ''))
            d_val = float(values.get('Demand', '0').replace(',', ''))
            s_val = float(values.get('Initial Inventory', '0').replace(',', ''))
        except Exception as e:
            raise ValueError(f'Numeric conversion error for record {i}: {e}')
        r_i[i] = r_val
        d_i[i] = d_val
        s_i[i] = s_val
    for i in ids:
        if i not in r_i or i not in d_i or i not in s_i:
            raise ValueError(f'Missing parameter for id {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(ids, vtype=GRB.INTEGER, lb=0, name='')
    for i in ids:
        model.addConstr(x[i] <= s_i[i], name=f'inv_{i}')
        model.addConstr(x[i] <= d_i[i], name=f'demand_{i}')
    objective = gurobipy.quicksum((r_i[i] * x[i] for i in ids))
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()