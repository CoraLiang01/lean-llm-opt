CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A smartphone retail outlet offers several models with revenue data provided in the ‘Revenue’ column. The '
          'retailer aims to maximize total revenue using the initial inventory of products classified under ‘TABLET’. '
          'Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities for these products are '
          'specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘TABLET’ smartphone model i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SmartphoneRetailOutletSalesData.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘TABLET’',
                                         'operator': 'prefix',
                                         'value': 'TABLET'}],
                         'logic': 'and'},
             'original_rows': 481,
             'records': [{'source_row': 454,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'TABLET_10084.74',
                                     'Revenue': '10084.74'}},
                         {'source_row': 455,
                          'values': {'Demand': '43',
                                     'Initial Inventory': '300',
                                     'Product Name': 'TABLET_12211.86',
                                     'Revenue': '12211.86'}},
                         {'source_row': 456,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'TABLET_14669.5',
                                     'Revenue': '14669.5'}},
                         {'source_row': 457,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'TABLET_14745.76',
                                     'Revenue': '14745.76'}},
                         {'source_row': 458,
                          'values': {'Demand': '20',
                                     'Initial Inventory': '100',
                                     'Product Name': 'TABLET_14754.24',
                                     'Revenue': '14754.24'}},
                         {'source_row': 459,
                          'values': {'Demand': '22',
                                     'Initial Inventory': '110',
                                     'Product Name': 'TABLET_16448.3',
                                     'Revenue': '16448.3'}},
                         {'source_row': 460,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': 'TABLET_16448.31',
                                     'Revenue': '16448.31'}},
                         {'source_row': 461,
                          'values': {'Demand': '16',
                                     'Initial Inventory': '80',
                                     'Product Name': 'TABLET_20143.22',
                                     'Revenue': '20143.22'}},
                         {'source_row': 462,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'TABLET_2042.38',
                                     'Revenue': '2042.38'}},
                         {'source_row': 463,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'TABLET_24915.25',
                                     'Revenue': '24915.25'}},
                         {'source_row': 464,
                          'values': {'Demand': '32',
                                     'Initial Inventory': '160',
                                     'Product Name': 'TABLET_24915.26',
                                     'Revenue': '24915.26'}},
                         {'source_row': 465,
                          'values': {'Demand': '14',
                                     'Initial Inventory': '70',
                                     'Product Name': 'TABLET_26448.3',
                                     'Revenue': '26448.3'}},
                         {'source_row': 466,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'TABLET_27042.38',
                                     'Revenue': '27042.38'}},
                         {'source_row': 467,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'TABLET_30000.0',
                                     'Revenue': '30000.0'}},
                         {'source_row': 468,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'TABLET_33397.46',
                                     'Revenue': '33397.46'}},
                         {'source_row': 469,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'TABLET_33398.3',
                                     'Revenue': '33398.3'}},
                         {'source_row': 470,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'TABLET_48567.8',
                                     'Revenue': '48567.8'}},
                         {'source_row': 471,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'TABLET_48644.07',
                                     'Revenue': '48644.07'}},
                         {'source_row': 472,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'TABLET_50262.72',
                                     'Revenue': '50262.72'}},
                         {'source_row': 473,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'TABLET_53736.44',
                                     'Revenue': '53736.44'}},
                         {'source_row': 474,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '40',
                                     'Product Name': 'TABLET_6957.62',
                                     'Revenue': '6957.62'}},
                         {'source_row': 475,
                          'values': {'Demand': '12',
                                     'Initial Inventory': '80',
                                     'Product Name': 'TABLET_6957.63',
                                     'Revenue': '6957.63'}},
                         {'source_row': 476,
                          'values': {'Demand': '60',
                                     'Initial Inventory': '300',
                                     'Product Name': 'TABLET_7550.84',
                                     'Revenue': '7550.84'}},
                         {'source_row': 477,
                          'values': {'Demand': '8',
                                     'Initial Inventory': '40',
                                     'Product Name': 'TABLET_7550.85',
                                     'Revenue': '7550.85'}},
                         {'source_row': 478,
                          'values': {'Demand': '8',
                                     'Initial Inventory': '40',
                                     'Product Name': 'TABLET_9584.74',
                                     'Revenue': '9584.74'}},
                         {'source_row': 479,
                          'values': {'Demand': '38',
                                     'Initial Inventory': '190',
                                     'Product Name': 'TABLET_9661.02',
                                     'Revenue': '9661.02'}},
                         {'source_row': 480,
                          'values': {'Demand': '4',
                                     'Initial Inventory': '20',
                                     'Product Name': 'TABLET_9669.5',
                                     'Revenue': '9669.5'}}],
             'returned_rows': 27,
             'role': 'decision, revenue, demand, inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for (source_row, row) in frame.iterrows():
        product_name = row['Product Name']
        if isinstance(product_name, str) and product_name.casefold().startswith('tablet'):
            I.append(product_name)
            try:
                revenue[product_name] = float(row['Revenue'])
            except Exception:
                raise ValueError(f"Invalid Revenue for {product_name}: {row['Revenue']}")
            try:
                demand[product_name] = int(float(row['Demand']))
            except Exception:
                raise ValueError(f"Invalid Demand for {product_name}: {row['Demand']}")
            try:
                inventory[product_name] = int(float(row['Initial Inventory']))
            except Exception:
                raise ValueError(f"Invalid Initial Inventory for {product_name}: {row['Initial Inventory']}")
    if not set(I) == set(revenue) == set(demand) == set(inventory):
        raise ValueError('Missing data for some TABLET products.')
    m = gp.Model('TABLET_Revenue_Maximization')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in I), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')