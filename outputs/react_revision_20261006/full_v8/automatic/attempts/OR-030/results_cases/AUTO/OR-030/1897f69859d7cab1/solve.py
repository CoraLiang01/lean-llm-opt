CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ '
          'column. The dealership aims to maximize total revenue using the initial inventory of car models classified '
          'under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are '
          'specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'BigMartSales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'car models classified under ‘FDK57’',
                                         'operator': 'prefix',
                                         'value': 'FDK57'}],
                         'logic': 'and'},
             'original_rows': 5681,
             'records': [{'source_row': 1163,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1501,
                          'values': {'Demand': '40',
                                     'Initial Inventory': '100',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.144'}},
                         {'source_row': 1576,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '200',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 1793,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.144'}},
                         {'source_row': 2438,
                          'values': {'Demand': '10',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.544'}},
                         {'source_row': 4297,
                          'values': {'Demand': '30',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '121.244'}},
                         {'source_row': 4806,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '250',
                                     'Product Name': 'FDK57',
                                     'Revenue': '119.744'}},
                         {'source_row': 4942,
                          'values': {'Demand': '50',
                                     'Initial Inventory': '150',
                                     'Product Name': 'FDK57',
                                     'Revenue': '120.844'}}],
             'returned_rows': 8,
             'role': 'car model revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES["file_0_view_0"]
    # Select all rows where "Product Name" has prefix 'FDK57'
    # (already filtered in the supplied frame)
    items = []
    revenue = {}
    demand = {}
    inventory = {}

    for source_row, row in frame.iterrows():
        product_name = row["Product Name"]
        # Only include rows with prefix 'FDK57'
        if isinstance(product_name, str) and product_name.casefold().startswith('fdk57'):
            key = (source_row, product_name)
            items.append(key)
            try:
                revenue[key] = float(row["Revenue"])
            except Exception:
                raise ValueError(f"Missing or invalid Revenue for row {source_row}")
            try:
                demand[key] = int(float(row["Demand"]))
            except Exception:
                raise ValueError(f"Missing or invalid Demand for row {source_row}")
            try:
                inventory[key] = int(float(row["Initial Inventory"]))
            except Exception:
                raise ValueError(f"Missing or invalid Initial Inventory for row {source_row}")

    # Validate all required data present
    if not (set(items) == set(revenue) == set(demand) == set(inventory)):
        raise ValueError("Mismatch in data dimensions for items, revenue, demand, or inventory.")

    m = gp.Model('FDK57_Car_Dealership')
    m.Params.MIPGap = 1e-4

    quantity_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='x')

    # Objective: maximize total revenue
    m.setObjective(gp.quicksum(revenue[i] * quantity_vars[i] for i in items), GRB.MAXIMIZE)

    # Inventory constraints: x_i <= I_i
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='inventory')

    # Demand constraints: x_i <= d_i
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='demand')

    m.optimize()
    return m

m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')