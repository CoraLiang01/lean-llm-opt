CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for '
          'these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant '
          'aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, '
          'which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, '
          'and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to '
          'be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The '
          'variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Aalop’',
                                         'operator': 'prefix',
                                         'value': 'Aalop'}],
                         'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1483',
                                     'Initial Inventory': '10440.0',
                                     'Product Name': 'Aalopuri',
                                     'Revenue': '20'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    df = CSVQA_FRAMES['file_0_view_0']
    mask = df['Product Name'].str.casefold().str.startswith('aalop')
    df_aalop = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_aalop.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(df_aalop['Product Name'])
    try:
        r = df_aalop.set_index('Product Name')['Revenue'].astype(float).to_dict()
        d = df_aalop.set_index('Product Name')['Demand'].astype(float).to_dict()
        s = df_aalop.set_index('Product Name')['Initial Inventory'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting parameters to numeric: {e}')
    for i in I:
        if i not in r or i not in d or i not in s:
            raise ValueError(f'Missing parameter for product {i}')
    var_ub = {i: min(d[i], s[i]) for i in I}
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, ub=var_ub, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((r[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(quantity_vars[i] <= d[i], name=f'demand_{i}')
        m.addConstr(quantity_vars[i] <= s[i], name=f'inventory_{i}')
        m.addConstr(quantity_vars[i] >= 0, name=f'nonneg_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)