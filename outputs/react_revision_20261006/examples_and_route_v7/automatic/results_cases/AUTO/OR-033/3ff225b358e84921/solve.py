CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Baby’',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    df = CSVQA_FRAMES['file_0_view_0']
    mask = df['Product Name'].str.casefold().str.startswith('baby')
    baby_df = df[mask]
    I = list(baby_df['Product Name'])
    if len(set(I)) != len(I):
        raise ValueError("Duplicate 'Product Name' entries found for 'Baby' products.")
    try:
        r_i = {row['Product Name']: float(row['Revenue']) for (_, row) in baby_df.iterrows()}
        d_i = {row['Product Name']: int(float(row['Demand'])) for (_, row) in baby_df.iterrows()}
        s_i = {row['Product Name']: int(float(row['Initial Inventory'])) for (_, row) in baby_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing numeric fields: {e}')
    for i in I:
        if i not in r_i or i not in d_i or i not in s_i:
            raise ValueError(f'Missing data for product {i}')
    m = gp.Model('Baby_Product_Revenue_Maximization')
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((r_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= s_i[i] for i in I), name='')
    m.addConstrs((quantity_vars[i] <= d_i[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)