CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Baby’',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES["file_0_view_0"]
    # Select all rows where "Product Name" starts with "Baby" (case-insensitive)
    baby_rows = []
    for source_row, row in frame.iterrows():
        pname = row["Product Name"]
        if isinstance(pname, str) and pname.casefold().startswith("baby"):
            baby_rows.append((source_row, row))
    if not baby_rows:
        raise ValueError("No products with 'Product Name' starting with 'Baby' found.")

    # Build index set I and parameter dictionaries
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for source_row, row in baby_rows:
        prod = row["Product Name"]
        try:
            A_i = float(row["Revenue"])
            d_i = float(row["Demand"])
            s_i = float(row["Initial Inventory"])
        except Exception as e:
            raise ValueError(f"Non-numeric value in required field for product '{prod}': {e}")
        I.append(prod)
        revenue[prod] = A_i
        demand[prod] = d_i
        inventory[prod] = s_i

    # Validate all required data present
    if not (set(revenue) == set(demand) == set(inventory) == set(I)):
        raise ValueError("Mismatch in index sets for revenue, demand, or inventory.")

    m = gp.Model("Baby_Product_Revenue_Maximization")
    m.Params.MIPGap = 1e-4

    # Decision variables: x_i >= 0, integer
    x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name="x")

    # Objective: maximize total revenue
    m.setObjective(gp.quicksum(revenue[i] * x_vars[i] for i in I), GRB.MAXIMIZE)

    # Constraints: x_i <= demand, x_i <= inventory
    m.addConstrs((x_vars[i] <= demand[i] for i in I), name="demand")
    m.addConstrs((x_vars[i] <= inventory[i] for i in I), name="inventory")

    m.optimize()
    return m

m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')