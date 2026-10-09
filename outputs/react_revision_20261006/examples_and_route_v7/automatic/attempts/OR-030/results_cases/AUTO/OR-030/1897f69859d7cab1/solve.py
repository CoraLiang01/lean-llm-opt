CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ '
          'column. The dealership aims to maximize total revenue using the initial inventory of car models classified '
          'under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are '
          'specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'BigMartSales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'car models classified under ‘FDK57’',
                                         'operator': 'prefix',
                                         'value': 'FDK57'}],
                         'logic': 'and'},
             'original_rows': 5681,
             'records': [{'source_row': 1163,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1501,
                          'values': {'Demand': '40',
                                     'Initial Inventory': '100',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1576,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 1793,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.144'}},
                         {'source_row': 2438,
                          'values': {'Demand': '10',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.544'}},
                         {'source_row': 4297,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 4806,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '250',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.744'}},
                         {'source_row': 4942,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.844'}}],
             'returned_rows': 8,
             'role': 'car model sales and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    table_id = 'file_0_view_0'
    df = CSVQA_FRAMES[table_id]
    mask = df['Product Name'].str.casefold().str.startswith('fdk57')
    filtered_df = df[mask].copy()
    if 'source_row' in filtered_df.columns:
        I = list(filtered_df['source_row'])
        filtered_df = filtered_df.set_index('source_row')
    else:
        I = list(filtered_df.index)

    def to_float(x):
        try:
            return float(x)
        except Exception:
            raise ValueError(f'Non-numeric value encountered: {x}')
    r = {}
    d = {}
    s = {}
    for i in I:
        row = filtered_df.loc[i]
        try:
            r[i] = to_float(row['Revenue'])
            d[i] = int(float(row['Demand']))
            s[i] = int(float(row['Initial Inventory']))
        except Exception as e:
            raise ValueError(f'Error parsing row {i}: {e}')
    if not set(I) == set(r) == set(d) == set(s):
        raise ValueError('Missing data for some car models.')
    m = gp.Model('Car_Dealership_FDK57')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((r[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= s[i] for i in I), name='')
    m.addConstrs((quantity_vars[i] <= d[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')