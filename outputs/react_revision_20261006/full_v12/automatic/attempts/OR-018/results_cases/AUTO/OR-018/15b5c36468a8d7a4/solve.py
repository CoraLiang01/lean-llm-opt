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
    baby_rows = []
    for (source_row, row) in frame.iterrows():
        pname = row['Product Name']
        if isinstance(pname, str) and pname.casefold().startswith('baby'):
            baby_rows.append((source_row, row))
    if not baby_rows:
        raise ValueError("No products with prefix 'Baby' found in 'Product Name'.")
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for (source_row, row) in baby_rows:
        identifier = row['Product Name']
        try:
            A_i = float(row['Revenue'])
            d_i = float(row['Demand'])
            s_i = float(row['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Non-numeric value in coefficients for product '{identifier}': {e}")
        I.append(identifier)
        revenue[identifier] = A_i
        demand[identifier] = d_i
        inventory[identifier] = s_i
    if not set(revenue) == set(demand) == set(inventory) == set(I):
        raise ValueError('Coefficient dictionaries do not cover the same set of products.')
    m = gp.Model('Baby_Product_Fulfillment')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= demand[i] for i in I), name='')
    m.addConstrs((x_vars[i] <= inventory[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()