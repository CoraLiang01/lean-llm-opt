CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Baby’',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product decision data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    df = CSVQA_FRAMES['file_0_view_0']
    mask = df['Product Name'].str.casefold().str.startswith('baby')
    baby_df = df[mask].copy()
    for col in ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']:
        if col not in baby_df.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(baby_df.index)
    product_names = baby_df['Product Name'].to_dict()
    try:
        r = baby_df['Revenue'].astype(float).to_dict()
        d = baby_df['Demand'].astype(float).to_dict()
        s = baby_df['Initial Inventory'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting numeric columns: {e}')
    ub = {i: min(d[i], s[i]) for i in I}
    for i in I:
        if ub[i] < 0:
            raise ValueError(f'Negative upper bound for product {product_names[i]} (row {i})')
    m = gp.Model('Baby_Product_Fulfillment')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(I, lb=0, ub=ub, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((r[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'x[{i}]: {quantity_vars[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)