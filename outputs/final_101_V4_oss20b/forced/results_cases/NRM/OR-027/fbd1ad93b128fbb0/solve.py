CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
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
    table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0'))
    records = table['records']
    indices = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        subcat = rec['values']['Sub Category']
        indices.append(subcat)
        revenue[subcat] = float(rec['values']['Revenue'])
        demand[subcat] = int(rec['values']['Demand'])
        inventory[subcat] = float(rec['values']['Initial Inventory'])
    for i in indices:
        if i not in revenue:
            raise ValueError(f'Missing revenue for {i}')
        if i not in demand:
            raise ValueError(f'Missing demand for {i}')
        if i not in inventory:
            raise ValueError(f'Missing inventory for {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(indices, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in indices)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= demand[i] for i in indices), name='')
    model.addConstrs((x[i] <= inventory[i] for i in indices), name='')
    model.optimize()
    return model
m = solve_problem()