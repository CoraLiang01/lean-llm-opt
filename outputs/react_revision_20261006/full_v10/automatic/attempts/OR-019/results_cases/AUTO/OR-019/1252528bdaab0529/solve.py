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
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘27in’',
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
             'role': 'product revenue, demand, and inventory for 27in products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    mask = frame['Product Name'].str.casefold().str.startswith('27in')
    selected = frame[mask]
    I = []
    A = {}
    d = {}
    I_inv = {}
    for (source_row, row) in selected.iterrows():
        prod = row['Product Name']
        try:
            revenue = float(row['Revenue'])
            demand = int(float(row['Demand']))
            inventory = int(float(row['Initial Inventory']))
        except Exception as e:
            raise ValueError(f"Non-numeric or missing data for product '{prod}': {e}")
        I.append(prod)
        A[prod] = revenue
        d[prod] = demand
        I_inv[prod] = inventory
    if not I:
        raise ValueError("No products found with 'Product Name' starting with '27in'.")
    for prod in I:
        if prod not in A or prod not in d or prod not in I_inv:
            raise ValueError(f"Missing parameter for product '{prod}'.")
    m = gp.Model('27in_Product_Revenue_Maximization')
    m.setParam('MIPGap', 0.0001)
    ub_dict = {prod: min(d[prod], I_inv[prod]) for prod in I}
    quantity_vars = m.addVars(I, lb=0, ub=ub_dict, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((A[prod] * quantity_vars[prod] for prod in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[prod] <= d[prod] for prod in I), name='')
    m.addConstrs((quantity_vars[prod] <= I_inv[prod] for prod in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)