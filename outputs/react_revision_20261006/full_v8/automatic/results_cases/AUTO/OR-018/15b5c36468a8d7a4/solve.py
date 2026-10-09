CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial '
          'Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ '
          'products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The '
          'decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Baby’',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory for Baby category',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    frame = CSVQA_FRAMES["file_0_view_0"]
    # Select all rows where Product Name has prefix "Baby" (already filtered in frame)
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for source_row, row in frame.iterrows():
        product_name = row["Product Name"]
        if not isinstance(product_name, str) or not product_name.casefold().startswith("baby"):
            continue
        items.append(product_name)
        try:
            revenue[product_name] = float(row["Revenue"])
        except Exception:
            raise ValueError(f"Missing or invalid Revenue for {product_name}")
        try:
            demand[product_name] = int(float(row["Demand"]))
        except Exception:
            raise ValueError(f"Missing or invalid Demand for {product_name}")
        try:
            inventory[product_name] = int(float(row["Initial Inventory"]))
        except Exception:
            raise ValueError(f"Missing or invalid Initial Inventory for {product_name}")

    if not items:
        raise ValueError("No eligible 'Baby' products found in the data.")

    # Validate all required coefficients are present
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f"Missing data for product {i}")

    m = gp.Model("baby_product_revenue")
    m.setParam("MIPGap", 1e-4)
    x_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name="x")

    m.setObjective(gp.quicksum(revenue[i] * x_vars[i] for i in items), GRB.MAXIMIZE)

    m.addConstrs((x_vars[i] <= demand[i] for i in items), name="demand")
    m.addConstrs((x_vars[i] <= inventory[i] for i in items), name="inventory")

    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m

m = solve_problem()