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
                                         'evidence': 'products classified under ‘Organ’',
                                         'operator': 'prefix',
                                         'value': 'Organ'}],
                         'logic': 'or'},
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
             'role': 'products with revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    organ_mask = frame['Sub Category'].str.casefold().str.startswith('organ')
    organ_frame = frame[organ_mask]
    required_columns = ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_columns:
        if col not in organ_frame.columns:
            raise ValueError(f'Missing required column: {col}')
    I = []
    A = {}
    d = {}
    s = {}
    for (source_row, row) in organ_frame.iterrows():
        identifier = row['Sub Category']
        try:
            revenue = float(row['Revenue'])
            demand = float(row['Demand'])
            inventory = float(row['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Non-numeric value in row {source_row}: {e}')
        key = (source_row, identifier)
        if key in I:
            raise ValueError(f'Duplicate key: {key}')
        I.append(key)
        A[key] = revenue
        d[key] = demand
        s[key] = inventory
    if not len(I) == len(A) == len(d) == len(s):
        raise ValueError('Coefficient dimension mismatch')
    m = gp.Model('Supermart_Organ_Revenue')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((A[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= d[i] for i in I), name='')
    m.addConstrs((quantity_vars[i] <= s[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')