CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store offers a variety of best-selling products with profit data provided in the ‘Revenue’ column. '
          'The retailer aims to maximize total revenue using the initial inventory of products classified under '
          '‘S700_’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified '
          'in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i '
          'indicate the number of units of each ‘S700_’ product i that the store intends to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SampleSalesData.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘S700_’',
                                         'operator': 'prefix',
                                         'value': 'S700_'}],
                         'logic': 'and'},
             'original_rows': 109,
             'records': [{'source_row': 95,
                          'values': {'Demand': '1219',
                                     'Initial Inventory': '9020',
                                     'Product Name': 'S700_1138',
                                     'Revenue': '70.67'}},
                         {'source_row': 96,
                          'values': {'Demand': '1127',
                                     'Initial Inventory': '8370',
                                     'Product Name': 'S700_1691',
                                     'Revenue': '100.0'}},
                         {'source_row': 97,
                          'values': {'Demand': '1129',
                                     'Initial Inventory': '8390',
                                     'Product Name': 'S700_1938',
                                     'Revenue': '70.15'}},
                         {'source_row': 98,
                          'values': {'Demand': '1176',
                                     'Initial Inventory': '8680',
                                     'Product Name': 'S700_2047',
                                     'Revenue': '100.0'}},
                         {'source_row': 99,
                          'values': {'Demand': '1301',
                                     'Initial Inventory': '9400',
                                     'Product Name': 'S700_2466',
                                     'Revenue': '100.0'}},
                         {'source_row': 100,
                          'values': {'Demand': '1340',
                                     'Initial Inventory': '9900',
                                     'Product Name': 'S700_2610',
                                     'Revenue': '65.77'}},
                         {'source_row': 101,
                          'values': {'Demand': '1357',
                                     'Initial Inventory': '9760',
                                     'Product Name': 'S700_2824',
                                     'Revenue': '100.0'}},
                         {'source_row': 102,
                          'values': {'Demand': '1158',
                                     'Initial Inventory': '8610',
                                     'Product Name': 'S700_2834',
                                     'Revenue': '100.0'}},
                         {'source_row': 103,
                          'values': {'Demand': '1287',
                                     'Initial Inventory': '9380',
                                     'Product Name': 'S700_3167',
                                     'Revenue': '74.4'}},
                         {'source_row': 104,
                          'values': {'Demand': '1281',
                                     'Initial Inventory': '9170',
                                     'Product Name': 'S700_3505',
                                     'Revenue': '81.14'}},
                         {'source_row': 105,
                          'values': {'Demand': '1135',
                                     'Initial Inventory': '8520',
                                     'Product Name': 'S700_3962',
                                     'Revenue': '100.0'}},
                         {'source_row': 106,
                          'values': {'Demand': '1392',
                                     'Initial Inventory': '10290',
                                     'Product Name': 'S700_4002',
                                     'Revenue': '61.44'}}],
             'returned_rows': 12,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    I = []
    A = {}
    d = {}
    s = {}
    for (source_row, row) in frame.iterrows():
        product_name = row['Product Name']
        if isinstance(product_name, str) and product_name.casefold().startswith('s700_'):
            I.append(product_name)
            try:
                A[product_name] = float(row['Revenue'])
            except Exception:
                raise ValueError(f'Missing or invalid Revenue for {product_name}')
            try:
                d[product_name] = int(float(row['Demand']))
            except Exception:
                raise ValueError(f'Missing or invalid Demand for {product_name}')
            try:
                s[product_name] = int(float(row['Initial Inventory']))
            except Exception:
                raise ValueError(f'Missing or invalid Initial Inventory for {product_name}')
    for i in I:
        if i not in A or i not in d or i not in s:
            raise ValueError(f'Missing data for product {i}')
    m = gp.Model('S700_Revenue_Maximization')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
    m.addConstrs((x_vars[i] <= s[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)