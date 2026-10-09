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
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Organ'}],
                         'logic': 'and'},
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
    mask = df['Sub Category'].str.casefold().str.startswith('organ')
    organ_df = df[mask].copy()
    required_cols = ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in organ_df.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(organ_df['Sub Category'])
    try:
        r = organ_df['Revenue'].astype(float)
        d = organ_df['Demand'].astype(float)
        s = organ_df['Initial Inventory'].astype(float)
    except Exception as e:
        raise ValueError(f'Error converting parameters to float: {e}')
    r_i = dict(zip(I, r))
    d_i = dict(zip(I, d))
    s_i = dict(zip(I, s))
    if not set(r_i.keys()) == set(d_i.keys()) == set(s_i.keys()) == set(I):
        raise ValueError('Parameter keys do not match index set I.')
    m = gp.Model('supermarket_organ_revenue')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, ub=None, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((r_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(quantity_vars[i] <= s_i[i], name='')
        m.addConstr(quantity_vars[i] <= d_i[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for v in quantity_vars.values():
            print(v.VarName, v.X)
    else:
        print(m.Status)
    return m
m = solve_problem(CSVQA_FRAMES)