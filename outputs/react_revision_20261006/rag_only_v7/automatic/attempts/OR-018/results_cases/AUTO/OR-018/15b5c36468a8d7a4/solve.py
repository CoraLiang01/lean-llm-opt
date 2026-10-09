CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
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
             'role': 'product demand, revenue, and inventory for Baby category',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    df = CSVQA_FRAMES['file_0_view_0']
    baby_mask = df['Product Name'].str.casefold().str.startswith('baby')
    baby_df = df[baby_mask].copy()
    I = list(baby_df['Product Name'])
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        if col not in baby_df.columns:
            raise ValueError(f'Missing required column: {col}')
        if baby_df[col].isnull().any() or (baby_df[col] == '').any():
            raise ValueError(f'Missing data in column: {col}')
    r = {}
    d = {}
    s = {}
    for (idx, row) in baby_df.iterrows():
        key = row['Product Name']
        try:
            r[key] = float(row['Revenue'])
            d[key] = int(row['Demand'])
            s[key] = int(row['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid numeric data for product {key}: {e}')
    quantity_vars = gp.Model('baby_fulfillment')
    x_vars = quantity_vars.addVars(I, lb=0, ub=[min(d[i], s[i]) for i in I], vtype=GRB.INTEGER, name='')
    quantity_vars.setObjective(gp.quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    quantity_vars.Params.MIPGap = 0.0001
    quantity_vars.optimize()
    if quantity_vars.Status == GRB.OPTIMAL:
        print(quantity_vars.ObjVal)
        for i in I:
            print(x_vars[i].VarName, x_vars[i].X)
    else:
        print(quantity_vars.Status)
    return quantity_vars
m = solve_problem(CSVQA_FRAMES)