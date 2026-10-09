CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'DifferentStoreSales.csv',
             'filters': {'conditions': [{'column': 'Product_Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Books’',
                                         'operator': 'prefix',
                                         'value': 'Books'}],
                         'logic': 'and'},
             'original_rows': 40,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1980',
                                     'Initial Inventory': '9920.0',
                                     'Product_Name': 'Books_15.15',
                                     'Revenue': '15.15'}},
                         {'source_row': 1,
                          'values': {'Demand': '3024',
                                     'Initial Inventory': '20160.0',
                                     'Product_Name': 'Books_30.3',
                                     'Revenue': '30.3'}},
                         {'source_row': 2,
                          'values': {'Demand': '4536',
                                     'Initial Inventory': '30000.0',
                                     'Product_Name': 'Books_45.45',
                                     'Revenue': '45.45'}},
                         {'source_row': 3,
                          'values': {'Demand': '5601',
                                     'Initial Inventory': '38360.0',
                                     'Product_Name': 'Books_60.6',
                                     'Revenue': '60.6'}},
                         {'source_row': 4,
                          'values': {'Demand': '7567',
                                     'Initial Inventory': '51450.0',
                                     'Product_Name': 'Books_75.75',
                                     'Revenue': '75.75'}}],
             'returned_rows': 5,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES["file_0_view_0"]
    # Select products where Product_Name has prefix "Books" (case-insensitive)
    books_mask = frame["Product_Name"].str.casefold().str.startswith("books")
    books_frame = frame[books_mask]
    # Index set I: all such products, preserve row order
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for source_row, row in books_frame.iterrows():
        product = row["Product_Name"]
        items.append(product)
        try:
            revenue[product] = float(row["Revenue"])
        except Exception:
            raise ValueError(f"Missing or invalid Revenue for {product}")
        try:
            demand[product] = int(float(row["Demand"]))
        except Exception:
            raise ValueError(f"Missing or invalid Demand for {product}")
        try:
            inventory[product] = int(float(row["Initial Inventory"]))
        except Exception:
            raise ValueError(f"Missing or invalid Initial Inventory for {product}")
    # Validate all items have coefficients
    if not (set(items) == set(revenue) == set(demand) == set(inventory)):
        raise ValueError("Mismatch in index sets for items and coefficients")
    m = gp.Model("Books_Revenue_Maximization")
    m.setParam("MIPGap", 1e-4)
    # Decision variables: x_i >= 0, integer
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name="x")
    # Objective: maximize sum_i A_i x_i
    m.setObjective(gp.quicksum(revenue[i] * quantity_vars[i] for i in items), GRB.MAXIMIZE)
    # Constraints: x_i <= d_i, x_i <= s_i
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name="demand")
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name="inventory")
    m.optimize()
    return m

m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')