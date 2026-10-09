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

def solve_problem():
    df = CSVQA_FRAMES['file_0_view_0']
    I = df.index.tolist()
    sub_categories = df['Sub Category'].tolist()
    try:
        revenue = {}
        demand = {}
        inventory = {}
        for (idx, row) in df.iterrows():
            key = row['Sub Category']
            try:
                revenue[key] = float(row['Revenue'])
            except Exception:
                raise ValueError(f"Invalid Revenue for {key}: {row['Revenue']}")
            try:
                demand[key] = int(float(row['Demand']))
            except Exception:
                raise ValueError(f"Invalid Demand for {key}: {row['Demand']}")
            try:
                inventory[key] = int(float(row['Initial Inventory']))
            except Exception:
                raise ValueError(f"Invalid Initial Inventory for {key}: {row['Initial Inventory']}")
    except Exception as e:
        raise ValueError(f'Error processing parameters: {e}')
    for key in sub_categories:
        if key not in revenue or key not in demand or key not in inventory:
            raise ValueError(f'Missing parameter for {key}')
    m = gp.Model('Supermart_Organ_Revenue')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(sub_categories, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in sub_categories)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in sub_categories), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in sub_categories), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()