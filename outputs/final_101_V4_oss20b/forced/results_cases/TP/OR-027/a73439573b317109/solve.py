CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'TP',
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
    data = CSVQA_DATA
    table = data['tables'][0]
    records = table['records']
    products = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec['values']
        subcat = values['Sub Category']
        products.append(subcat)
        try:
            revenue[subcat] = float(values['Revenue'])
            demand[subcat] = float(values['Demand'])
            inventory[subcat] = float(values['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing field {e} in record {rec}')
    for p in products:
        if p not in revenue or p not in demand or p not in inventory:
            raise ValueError(f'Missing parameter for {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in products), name='')
    m.addConstrs((x[i] <= inventory[i] for i in products), name='')
    m.optimize()
    return m
m = solve_problem()