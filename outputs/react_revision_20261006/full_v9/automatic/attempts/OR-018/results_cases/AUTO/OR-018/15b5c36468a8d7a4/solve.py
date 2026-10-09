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
             'role': 'product demand, revenue, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    frame = CSVQA_FRAMES['file_0_view_0']
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for (source_row, row) in frame.iterrows():
        product_name = row['Product Name']
        if not isinstance(product_name, str) or not product_name.casefold().startswith('baby'):
            continue
        items.append(product_name)
        try:
            revenue[product_name] = float(row['Revenue'])
        except Exception:
            raise ValueError(f'Missing or invalid Revenue for {product_name}')
        try:
            demand[product_name] = int(float(row['Demand']))
        except Exception:
            raise ValueError(f'Missing or invalid Demand for {product_name}')
        try:
            inventory[product_name] = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f'Missing or invalid Initial Inventory for {product_name}')
    if not set(items) == set(revenue) == set(demand) == set(inventory):
        raise ValueError('Mismatch in product indices or missing data.')
    upper_bounds = {i: min(demand[i], inventory[i]) for i in items}
    m = gp.Model('Baby_Product_Fulfillment')
    quantity_vars = m.addVars(items, lb=0, ub=[upper_bounds[i] for i in items], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()