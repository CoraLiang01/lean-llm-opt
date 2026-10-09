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
                                         'inclusive': 'both',
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

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES["file_0_view_0"]
    # Select products where Product Name has prefix "Aalop" (case-insensitive)
    aalop_mask = frame["Product Name"].str.casefold().str.startswith("aalop")
    aalop_frame = frame[aalop_mask]
    # Build index set I
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for source_row, row in aalop_frame.iterrows():
        product = row["Product Name"]
        try:
            A_i = float(row["Revenue"])
            d_i = float(row["Demand"])
            s_i = float(row["Initial Inventory"])
        except Exception as e:
            raise ValueError(f"Non-numeric coefficient for product '{product}': {e}")
        items.append(product)
        revenue[product] = A_i
        demand[product] = d_i
        inventory[product] = s_i

    # Validate that all required data is present
    if not items:
        raise ValueError("No products with prefix 'Aalop' found in 'Product Name' column.")

    m = gp.Model("Aalop_Revenue_Maximization")
    m.setParam("MIPGap", 1e-4)
    # Decision variables: x_i >= 0, continuous
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.CONTINUOUS, name="x")
    # Objective: maximize total revenue
    m.setObjective(gp.quicksum(revenue[i] * quantity_vars[i] for i in items), GRB.MAXIMIZE)
    # Demand constraints: x_i <= d_i
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name="demand")
    # Inventory constraints: x_i <= s_i
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name="inventory")
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m

m = solve_problem(CSVQA_FRAMES)