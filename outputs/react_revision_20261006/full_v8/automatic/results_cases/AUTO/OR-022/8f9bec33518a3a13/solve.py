CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 0,
                          'values': {'Demand': '8230',
                                     'Initial Inventory': '41290',
                                     'Product Name': '20in Monitor',
                                     'Revenue': '38.4965'}},
                         {'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '261.2933'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965'}},
                         {'source_row': 3,
                          'values': {'Demand': '12380',
                                     'Initial Inventory': '61990',
                                     'Product Name': '34in Ultrawide Monitor',
                                     'Revenue': '254.5933'}},
                         {'source_row': 4,
                          'values': {'Demand': '49137',
                                     'Initial Inventory': '276350',
                                     'Product Name': 'AA Batteries (4-pack)',
                                     'Revenue': '1.92'}},
                         {'source_row': 5,
                          'values': {'Demand': '53353',
                                     'Initial Inventory': '310170',
                                     'Product Name': 'AAA Batteries (4-pack)',
                                     'Revenue': '1.495'}},
                         {'source_row': 6,
                          'values': {'Demand': '31211',
                                     'Initial Inventory': '156610',
                                     'Product Name': 'Apple Airpods Headphones',
                                     'Revenue': '52.5'}},
                         {'source_row': 7,
                          'values': {'Demand': '26782',
                                     'Initial Inventory': '134570',
                                     'Product Name': 'Bose SoundSport Headphones',
                                     'Revenue': '49.995'}},
                         {'source_row': 8,
                          'values': {'Demand': '9619',
                                     'Initial Inventory': '48190',
                                     'Product Name': 'Flatscreen TV',
                                     'Revenue': '201.0'}},
                         {'source_row': 9,
                          'values': {'Demand': '11057',
                                     'Initial Inventory': '55320',
                                     'Product Name': 'Google Phone',
                                     'Revenue': '402.0'}},
                         {'source_row': 10,
                          'values': {'Demand': '1292',
                                     'Initial Inventory': '6460',
                                     'Product Name': 'LG Dryer',
                                     'Revenue': '402.0'}},
                         {'source_row': 11,
                          'values': {'Demand': '1332',
                                     'Initial Inventory': '6660',
                                     'Product Name': 'LG Washing Machine',
                                     'Revenue': '402.0'}},
                         {'source_row': 12,
                          'values': {'Demand': '44932',
                                     'Initial Inventory': '232170',
                                     'Product Name': 'Lightning Charging Cable',
                                     'Revenue': '7.475'}},
                         {'source_row': 13,
                          'values': {'Demand': '9452',
                                     'Initial Inventory': '47280',
                                     'Product Name': 'Macbook Pro Laptop',
                                     'Revenue': '1139.0'}},
                         {'source_row': 14,
                          'values': {'Demand': '8258',
                                     'Initial Inventory': '41300',
                                     'Product Name': 'ThinkPad Laptop',
                                     'Revenue': '669.9933'}},
                         {'source_row': 15,
                          'values': {'Demand': '45987',
                                     'Initial Inventory': '239750',
                                     'Product Name': 'USB-C Charging Cable',
                                     'Revenue': '5.975'}},
                         {'source_row': 16,
                          'values': {'Demand': '4133',
                                     'Initial Inventory': '20680',
                                     'Product Name': 'Vareebadd Phone',
                                     'Revenue': '268.0'}},
                         {'source_row': 17,
                          'values': {'Demand': '39525',
                                     'Initial Inventory': '205570',
                                     'Product Name': 'Wired Headphones',
                                     'Revenue': '11.99'}},
                         {'source_row': 18,
                          'values': {'Demand': '13691',
                                     'Initial Inventory': '68490',
                                     'Product Name': 'iPhone',
                                     'Revenue': '469.0'}}],
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

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES["file_0_view_0"]
    # Select products whose "Product Name" contains "27in" (case-insensitive)
    mask = frame["Product Name"].str.casefold().str.contains("27in")
    selected = frame[mask]
    # Build index set I: use row index as unique key, but keep "Product Name" for reporting
    I = list(selected.index)
    if len(I) == 0:
        raise ValueError("No products found with '27in' in 'Product Name'.")
    # Extract parameters
    A = {}
    d = {}
    s = {}
    for idx, row in selected.iterrows():
        try:
            A[idx] = float(row["Revenue"])
        except Exception:
            raise ValueError(f"Non-numeric Revenue for row {idx}: {row['Revenue']}")
        try:
            d[idx] = int(float(row["Demand"]))
        except Exception:
            raise ValueError(f"Non-numeric Demand for row {idx}: {row['Demand']}")
        try:
            s[idx] = int(float(row["Initial Inventory"]))
        except Exception:
            raise ValueError(f"Non-numeric Initial Inventory for row {idx}: {row['Initial Inventory']}")
    # Validate all indices have all parameters
    for idx in I:
        if idx not in A or idx not in d or idx not in s:
            raise ValueError(f"Missing parameter for index {idx}")
    m = gp.Model("27in_Product_Revenue_Maximization")
    m.setParam("MIPGap", 1e-4)
    # Decision variables: x_i >= 0, integer
    x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name="x")
    # Objective: maximize sum_i A_i x_i
    m.setObjective(gp.quicksum(A[i] * x_vars[i] for i in I), GRB.MAXIMIZE)
    # Constraints: x_i <= s_i (inventory), x_i <= d_i (demand)
    m.addConstrs((x_vars[i] <= s[i] for i in I), name="inventory")
    m.addConstrs((x_vars[i] <= d[i] for i in I), name="demand")
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'x[{i}] ({selected.loc[i, "Product Name"]}): {x_vars[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m

m = solve_problem(CSVQA_FRAMES)