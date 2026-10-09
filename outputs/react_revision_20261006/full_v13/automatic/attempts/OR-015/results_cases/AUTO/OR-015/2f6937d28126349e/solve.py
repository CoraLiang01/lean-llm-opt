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
    frame = CSVQA_FRAMES['file_0_view_0']
    aalop_rows = []
    for (source_row, row) in frame.iterrows():
        pname = row['Product Name']
        if isinstance(pname, str) and pname.casefold().startswith('aalop'):
            aalop_rows.append((source_row, row))
    if not aalop_rows:
        raise ValueError("No products with prefix 'Aalop' found in 'Product Name'.")
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for (source_row, row) in aalop_rows:
        pname = row['Product Name']
        items.append(pname)
        try:
            revenue[pname] = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Invalid Revenue for product '{pname}' (row {source_row}): {row['Revenue']}")
        try:
            demand[pname] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Invalid Demand for product '{pname}' (row {source_row}): {row['Demand']}")
        try:
            inventory[pname] = float(row['Initial Inventory'])
        except Exception:
            raise ValueError(f"Invalid Initial Inventory for product '{pname}' (row {source_row}): {row['Initial Inventory']}")
    if not set(revenue) == set(demand) == set(inventory) == set(items):
        raise ValueError('Mismatch in parameter keys for items.')
    m = gp.Model('Aalop_Revenue_Maximization')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')