CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘27in’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}}],
             'returned_rows': 2,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    df = CSVQA_FRAMES['file_0_view_0']
    I = df.index.tolist()
    product_names = df['Product Name'].tolist()
    try:
        r_i = {i: float(df.at[i, 'Revenue']) for i in I}
        d_i = {i: int(df.at[i, 'Demand']) for i in I}
        s_i = {i: int(df.at[i, 'Initial Inventory']) for i in I}
    except Exception as e:
        raise ValueError(f'Error parsing numeric fields: {e}')
    if not len(r_i) == len(d_i) == len(s_i) == len(I):
        raise ValueError('Missing data for some products.')
    m = gp.Model('27in_Product_Revenue_Maximization')
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((r_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= d_i[i] for i in I), name='')
    m.addConstrs((quantity_vars[i] <= s_i[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'x[{product_names[I.index(i)]}]: {quantity_vars[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)