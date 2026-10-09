CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'This supermarket offers a variety of best-selling products, and the specific revenue information is shown '
          'in the table, with relevant data provided in the “Revenue” column. Each product has its own demand level. '
          'The retailer’s objective is to maximize the total revenue by focusing on the sales volume of “4U” products. '
          'The initial inventory levels of these products are detailed in the “Initial Inventory” column. During the '
          'sales period, restocking is not allowed and there is no in-transit inventory.\n'
          '\n'
          'Demand for the “4U” products during the sales horizon is assumed to be deterministic and known in advance, '
          'with demand quantities specified in the “Demand” column. The decision variable x_i represents the number of '
          'units of each “4U” product i that the retailer plans to fulfill, where each x_i is a non-negative integer. '
          'Because the fulfillment quantity cannot exceed either the available inventory or the realized demand, the '
          'decision variables must satisfy both inventory and demand constraints.\n'
          '\n'
          'The retailer therefore aims to determine the optimal fulfillment quantities in order to maximize total '
          'revenue while respecting both inventory availability and demand limits.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineSalesinUSA.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'sales volume of “4U” products',
                                         'operator': 'prefix',
                                         'value': '4U'}],
                         'logic': 'and'},
             'original_rows': 47932,
             'records': [{'source_row': 1,
                          'values': {'Demand': '5',
                                     'Initial Inventory': '30',
                                     'Product Name': '4U_Service_22',
                                     'Revenue': '56.0'}},
                         {'source_row': 2,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': '4U_Service_36',
                                     'Revenue': '21.6'}},
                         {'source_row': 3,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': '4U_Service_7',
                                     'Revenue': '62.5'}}],
             'returned_rows': 3,
             'role': 'revenue management product data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
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
        product_name = row['Product Name']
        if not isinstance(product_name, str) or not product_name.casefold().startswith('4u'):
            continue
        products.append(product_name)
        try:
            revenue[product_name] = float(row['Revenue'])
        except Exception:
            raise ValueError(f'Missing or invalid Revenue for product {product_name}')
        try:
            demand[product_name] = int(float(row['Demand']))
        except Exception:
            raise ValueError(f'Missing or invalid Demand for product {product_name}')
        try:
            inventory[product_name] = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f'Missing or invalid Initial Inventory for product {product_name}')
    for p in products:
        if p not in revenue or p not in demand or p not in inventory:
            raise ValueError(f'Missing data for product {p}')
    m = gp.Model('4U_Product_Revenue_Maximization')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[p] <= inventory[p] for p in products), name='')
    m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')