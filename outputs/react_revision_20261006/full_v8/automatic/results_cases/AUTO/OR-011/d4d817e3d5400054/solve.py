CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the '
          '‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective '
          'is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. '
          'The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. '
          'During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n'
          '\n'
          'Demand for each product during the sales period is assumed to be deterministic and known in advance, with '
          'demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of '
          'units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative '
          'integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the '
          'fulfillment quantities must satisfy both inventory and demand constraints.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘id999’',
                                         'operator': 'prefix',
                                         'value': 'id999'}],
                         'logic': 'and'},
             'original_rows': 900,
             'records': [{'source_row': 899,
                          'values': {'Demand': '8171',
                                     'Initial Inventory': '56450',
                                     'Revenue': '434.74',
                                     'id_number': 'id999'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES["file_0_view_0"]
    # Select all rows where id_number has prefix "id999" (case-insensitive)
    mask = frame["id_number"].str.casefold().str.startswith("id999")
    filtered = frame[mask]
    # Build index set I and parameter dictionaries
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for source_row, row in filtered.iterrows():
        identifier = row["id_number"]
        try:
            A_i = float(row["Revenue"])
            d_i = float(row["Demand"])
            s_i = float(row["Initial Inventory"])
        except Exception as e:
            raise ValueError(f"Non-numeric parameter for product {identifier}: {e}")
        I.append(identifier)
        revenue[identifier] = A_i
        demand[identifier] = d_i
        inventory[identifier] = s_i
    # Validate that all required data is present
    if not I:
        raise ValueError("No products with id_number prefix 'id999' found in the data.")
    if set(revenue.keys()) != set(I) or set(demand.keys()) != set(I) or set(inventory.keys()) != set(I):
        raise ValueError("Missing parameter data for some products in index set I.")
    m = gp.Model("Supermarket_Revenue_Maximization")
    m.setParam("MIPGap", 1e-4)
    # Decision variables: x_i >= 0, integer
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name="x")
    # Objective: maximize total revenue
    m.setObjective(gp.quicksum(revenue[i] * quantity_vars[i] for i in I), GRB.MAXIMIZE)
    # Inventory constraints: x_i <= s_i
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in I), name="inventory")
    # Demand constraints: x_i <= d_i
    m.addConstrs((quantity_vars[i] <= demand[i] for i in I), name="demand")
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m

m = solve_problem(CSVQA_FRAMES)