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
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1483',
                                     'Initial Inventory': '10440.0',
                                     'Product Name': 'Aalopuri',
                                     'Revenue': '20'}},
                         {'source_row': 1,
                          'values': {'Demand': '1918',
                                     'Initial Inventory': '13610.0',
                                     'Product Name': 'Cold coffee',
                                     'Revenue': '40'}},
                         {'source_row': 2,
                          'values': {'Demand': '1623',
                                     'Initial Inventory': '11500.0',
                                     'Product Name': 'Frankie',
                                     'Revenue': '50'}},
                         {'source_row': 3,
                          'values': {'Demand': '1720',
                                     'Initial Inventory': '12260.0',
                                     'Product Name': 'Panipuri',
                                     'Revenue': '20'}},
                         {'source_row': 4,
                          'values': {'Demand': '1558',
                                     'Initial Inventory': '10970.0',
                                     'Product Name': 'Sandwich',
                                     'Revenue': '60'}},
                         {'source_row': 5,
                          'values': {'Demand': '1791',
                                     'Initial Inventory': '12780.0',
                                     'Product Name': 'Sugarcane juice',
                                     'Revenue': '25'}},
                         {'source_row': 6,
                          'values': {'Demand': '1426',
                                     'Initial Inventory': '10060.0',
                                     'Product Name': 'Vadapav',
                                     'Revenue': '20'}}],
             'returned_rows': 7,
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
    frame = CSVQA_FRAMES['file_0_view_0']
    products = []
    revenue = {}
    demand = {}
    inventory = {}
    for (source_row, row) in frame.iterrows():
        product = row['Product Name']
        products.append(product)
        try:
            revenue[product] = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Invalid Revenue for product {product}: {row['Revenue']}")
        try:
            demand[product] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Invalid Demand for product {product}: {row['Demand']}")
        try:
            inventory[product] = float(row['Initial Inventory'])
        except Exception:
            raise ValueError(f"Invalid Initial Inventory for product {product}: {row['Initial Inventory']}")
    if not set(revenue) == set(demand) == set(inventory) == set(products):
        raise ValueError('Mismatch in product indices among parameters.')
    m = gp.Model('Aalop_Inventory_Fulfillment')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
    m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)