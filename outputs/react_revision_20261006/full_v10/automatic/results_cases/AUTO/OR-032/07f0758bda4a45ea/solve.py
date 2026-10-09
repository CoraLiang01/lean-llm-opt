CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'DifferentStoreSales.csv',
             'filters': {'conditions': [{'column': 'Product_Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Books’',
                                         'operator': 'prefix',
                                         'value': 'Books'}],
                         'logic': 'and'},
             'original_rows': 40,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1980',
                                     'Initial Inventory': '9920.0',
                                     'Product_Name': 'Books_15.15',
                                     'Revenue': '15.15'}},
                         {'source_row': 1,
                          'values': {'Demand': '3024',
                                     'Initial Inventory': '20160.0',
                                     'Product_Name': 'Books_30.3',
                                     'Revenue': '30.3'}},
                         {'source_row': 2,
                          'values': {'Demand': '4536',
                                     'Initial Inventory': '30000.0',
                                     'Product_Name': 'Books_45.45',
                                     'Revenue': '45.45'}},
                         {'source_row': 3,
                          'values': {'Demand': '5601',
                                     'Initial Inventory': '38360.0',
                                     'Product_Name': 'Books_60.6',
                                     'Revenue': '60.6'}},
                         {'source_row': 4,
                          'values': {'Demand': '7567',
                                     'Initial Inventory': '51450.0',
                                     'Product_Name': 'Books_75.75',
                                     'Revenue': '75.75'}}],
             'returned_rows': 5,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    books_rows = []
    for (source_row, row) in frame.iterrows():
        pname = row['Product_Name']
        if isinstance(pname, str) and pname.casefold().startswith('books'):
            books_rows.append((source_row, row))
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for (source_row, row) in books_rows:
        pname = row['Product_Name']
        try:
            A_i = float(row['Revenue'])
            d_i = int(float(row['Demand']))
            I_i = int(float(row['Initial Inventory']))
        except Exception as e:
            raise ValueError(f'Invalid numeric data for product {pname}: {e}')
        items.append(pname)
        revenue[pname] = A_i
        demand[pname] = d_i
        inventory[pname] = I_i
    if not items:
        raise ValueError("No products classified under 'Books' found in file_0_view_0.")
    for pname in items:
        if pname not in revenue or pname not in demand or pname not in inventory:
            raise ValueError(f'Missing parameter for product {pname}.')
    m = gp.Model('Books_Revenue_Maximization')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in x_vars.values():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)