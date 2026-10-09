CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 0,
                          'values': {'Demand': '8230',
                                     'Initial Inventory': '41290',
                                     'Product Name': '20in Monitor',
                                     'Revenue': '109.99'}},
                         {'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}},
                         {'source_row': 3,
                          'values': {'Demand': '12380',
                                     'Initial Inventory': '61990',
                                     'Product Name': '34in Ultrawide Monitor',
                                     'Revenue': '379.99'}},
                         {'source_row': 4,
                          'values': {'Demand': '49129',
                                     'Initial Inventory': '276350',
                                     'Product Name': 'AA Batteries (4-pack)',
                                     'Revenue': '3.84'}},
                         {'source_row': 5,
                          'values': {'Demand': '53317',
                                     'Initial Inventory': '310170',
                                     'Product Name': 'AAA Batteries (4-pack)',
                                     'Revenue': '2.99'}},
                         {'source_row': 6,
                          'values': {'Demand': '31210',
                                     'Initial Inventory': '156610',
                                     'Product Name': 'Apple Airpods Headphones',
                                     'Revenue': '150.0'}},
                         {'source_row': 7,
                          'values': {'Demand': '26784',
                                     'Initial Inventory': '134570',
                                     'Product Name': 'Bose SoundSport Headphones',
                                     'Revenue': '99.99'}},
                         {'source_row': 8,
                          'values': {'Demand': '9619',
                                     'Initial Inventory': '48190',
                                     'Product Name': 'Flatscreen TV',
                                     'Revenue': '300.0'}},
                         {'source_row': 9,
                          'values': {'Demand': '11057',
                                     'Initial Inventory': '55320',
                                     'Product Name': 'Google Phone',
                                     'Revenue': '600.0'}},
                         {'source_row': 10,
                          'values': {'Demand': '1292',
                                     'Initial Inventory': '6460',
                                     'Product Name': 'LG Dryer',
                                     'Revenue': '600.0'}},
                         {'source_row': 11,
                          'values': {'Demand': '1332',
                                     'Initial Inventory': '6660',
                                     'Product Name': 'LG Washing Machine',
                                     'Revenue': '600.0'}},
                         {'source_row': 12,
                          'values': {'Demand': '44936',
                                     'Initial Inventory': '232170',
                                     'Product Name': 'Lightning Charging Cable',
                                     'Revenue': '14.95'}},
                         {'source_row': 13,
                          'values': {'Demand': '9452',
                                     'Initial Inventory': '47280',
                                     'Product Name': 'Macbook Pro Laptop',
                                     'Revenue': '1700.0'}},
                         {'source_row': 14,
                          'values': {'Demand': '8258',
                                     'Initial Inventory': '41300',
                                     'Product Name': 'ThinkPad Laptop',
                                     'Revenue': '999.99'}},
                         {'source_row': 15,
                          'values': {'Demand': '45977',
                                     'Initial Inventory': '239750',
                                     'Product Name': 'USB-C Charging Cable',
                                     'Revenue': '11.95'}},
                         {'source_row': 16,
                          'values': {'Demand': '4133',
                                     'Initial Inventory': '20680',
                                     'Product Name': 'Vareebadd Phone',
                                     'Revenue': '400.0'}},
                         {'source_row': 17,
                          'values': {'Demand': '39520',
                                     'Initial Inventory': '205570',
                                     'Product Name': 'Wired Headphones',
                                     'Revenue': '11.99'}},
                         {'source_row': 18,
                          'values': {'Demand': '13691',
                                     'Initial Inventory': '68490',
                                     'Product Name': 'iPhone',
                                     'Revenue': '700.0'}}],
             'returned_rows': 19,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': 'expected a string or tuple, not list',
                'planner_errors': ['expected a string or tuple, not list'],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    table_id = 'file_0_view_0'
    df = CSVQA_FRAMES[table_id]
    mask = df['Product Name'].str.casefold().str.contains('27in')
    df_27in = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_27in.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(df_27in.index)
    r_i = {}
    d_i = {}
    s_i = {}
    for idx in I:
        row = df_27in.loc[idx]
        try:
            r = float(row['Revenue'])
            d = int(float(row['Demand']))
            s = int(float(row['Initial Inventory']))
        except Exception as e:
            raise ValueError(f'Invalid numeric data in row {idx}: {e}')
        r_i[idx] = r
        d_i[idx] = d
        s_i[idx] = s
    if set(r_i.keys()) != set(I) or set(d_i.keys()) != set(I) or set(s_i.keys()) != set(I):
        raise ValueError('Parameter coverage mismatch with index set I')
    m = gp.Model()
    quantity_vars = m.addVars(I, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='')
    for idx in I:
        m.addConstr(quantity_vars[idx] <= d_i[idx], name=f'demand_{idx}')
        m.addConstr(quantity_vars[idx] <= s_i[idx], name=f'inventory_{idx}')
    m.setObjective(gp.quicksum((r_i[idx] * quantity_vars[idx] for idx in I)), GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for idx in I:
            print(quantity_vars[idx].VarName, quantity_vars[idx].X)
    else:
        print(m.Status)
    return m
m = solve_problem(CSVQA_FRAMES)