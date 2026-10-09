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

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES["file_0_view_0"]
    # Select all rows where Sub Category has prefix "Organ" (case-insensitive)
    organ_mask = frame["Sub Category"].str.casefold().str.startswith("organ")
    organ_frame = frame[organ_mask]
    # Build index set I: use row indices as unique keys
    I = list(organ_frame.index)
    # Extract parameters for each i in I
    A = {}
    d = {}
    s = {}
    for i in I:
        row = organ_frame.loc[i]
        try:
            A[i] = float(row["Revenue"])
        except Exception:
            raise ValueError(f"Missing or invalid Revenue for row {i}")
        try:
            d[i] = float(row["Demand"])
        except Exception:
            raise ValueError(f"Missing or invalid Demand for row {i}")
        try:
            s[i] = float(row["Initial Inventory"])
        except Exception:
            raise ValueError(f"Missing or invalid Initial Inventory for row {i}")
    # Validate all required data present
    if not (set(A) == set(d) == set(s) == set(I)):
        raise ValueError("Mismatch in parameter keys for index set I")
    m = gp.Model("Organ_Product_Revenue_Maximization")
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name="x")
    m.setObjective(gp.quicksum(A[i] * x_vars[i] for i in I), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= d[i] for i in I), name="demand")
    m.addConstrs((x_vars[i] <= s[i] for i in I), name="inventory")
    m.Params.MIPGap = 1e-4
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m

m = solve_problem(CSVQA_FRAMES)