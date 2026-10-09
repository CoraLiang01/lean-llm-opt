CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'An e-commerce platform sells women’s clothing items with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘FAUX’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent '
          'the number of units of each ‘FAUX’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'ZARASales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘FAUX’',
                                         'operator': 'prefix',
                                         'value': 'FAUX'}],
                         'logic': 'and'},
             'original_rows': 194,
             'records': [{'source_row': 56,
                          'values': {'Demand': '3025',
                                     'Initial Inventory': '20970',
                                     'Product Name': 'FAUX FUR JEWEL SWEATER',
                                     'Revenue': '35.9'}},
                         {'source_row': 57,
                          'values': {'Demand': '9585',
                                     'Initial Inventory': '71970',
                                     'Product Name': 'FAUX LEATHER BOMBER JACKET',
                                     'Revenue': '69.9'}},
                         {'source_row': 58,
                          'values': {'Demand': '4486',
                                     'Initial Inventory': '32730',
                                     'Product Name': 'FAUX LEATHER BOXY FIT JACKET',
                                     'Revenue': '99.9'}},
                         {'source_row': 59,
                          'values': {'Demand': '10322',
                                     'Initial Inventory': '71130',
                                     'Product Name': 'FAUX LEATHER JACKET',
                                     'Revenue': '99.9'}},
                         {'source_row': 60,
                          'values': {'Demand': '4868',
                                     'Initial Inventory': '34910',
                                     'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION',
                                     'Revenue': '159.0'}},
                         {'source_row': 61,
                          'values': {'Demand': '8482',
                                     'Initial Inventory': '64010',
                                     'Product Name': 'FAUX LEATHER PUFFER JACKET',
                                     'Revenue': '69.99'}},
                         {'source_row': 62,
                          'values': {'Demand': '2607',
                                     'Initial Inventory': '20760',
                                     'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS',
                                     'Revenue': '99.9'}},
                         {'source_row': 63,
                          'values': {'Demand': '1784',
                                     'Initial Inventory': '12490',
                                     'Product Name': 'FAUX SHEARLING PLAID JACKET',
                                     'Revenue': '89.9'}},
                         {'source_row': 64,
                          'values': {'Demand': '6626',
                                     'Initial Inventory': '50300',
                                     'Product Name': 'FAUX SUEDE BOMBER JACKET',
                                     'Revenue': '69.9'}},
                         {'source_row': 65,
                          'values': {'Demand': '3256',
                                     'Initial Inventory': '24570',
                                     'Product Name': 'FAUX SUEDE JACKET',
                                     'Revenue': '89.9'}},
                         {'source_row': 66,
                          'values': {'Demand': '2955',
                                     'Initial Inventory': '24430',
                                     'Product Name': 'FAUX SUEDE OVERSHIRT',
                                     'Revenue': '69.9'}},
                         {'source_row': 67,
                          'values': {'Demand': '910',
                                     'Initial Inventory': '7070',
                                     'Product Name': 'FAUX SUEDE PATCH JACKET',
                                     'Revenue': '89.9'}}],
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
    faux_rows = []
    for (source_row, row) in frame.iterrows():
        product_name = row['Product Name']
        if isinstance(product_name, str) and product_name.casefold().startswith('faux'):
            faux_rows.append((product_name, row))
    if not faux_rows:
        raise ValueError("No products with names starting with 'FAUX' found in file_0_view_0.")
    I = []
    A = {}
    d = {}
    s = {}
    for (product_name, row) in faux_rows:
        try:
            revenue = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Invalid Revenue for product '{product_name}': {row['Revenue']}")
        try:
            demand = float(row['Demand'])
        except Exception:
            raise ValueError(f"Invalid Demand for product '{product_name}': {row['Demand']}")
        try:
            inventory = float(row['Initial Inventory'])
        except Exception:
            raise ValueError(f"Invalid Initial Inventory for product '{product_name}': {row['Initial Inventory']}")
        I.append(product_name)
        A[product_name] = revenue
        d[product_name] = demand
        s[product_name] = inventory
    m = gp.Model('ZARA_FAUX_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= s[i] for i in I), name='')
    m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')