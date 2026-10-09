CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail establishment seeks to optimize the allocation of its merchandise across distinct categories '
          '(electronics, apparel, homeware, etc.) to achieve the highest possible total revenue. Each category '
          'maintains its own demand level, with revenue figures provided in the ‘Revenue’ column and current stock '
          'quantities documented in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The optimization challenge involves '
          'determining the ideal fulfillment quantities x_i for each product that allocate available inventory while '
          'respecting inventory limits and maximizing total revenue.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailSalesDataset.csv',
             'filters': {},
             'original_rows': 35,
             'records': [{'source_row': 0,
                          'values': {'Demand': '240',
                                     'Initial Inventory': '1570',
                                     'Product Name': 'Beauty - 25',
                                     'Revenue': '25'}},
                         {'source_row': 1,
                          'values': {'Demand': '202',
                                     'Initial Inventory': '1330',
                                     'Product Name': 'Beauty - 30',
                                     'Revenue': '30'}},
                         {'source_row': 2,
                          'values': {'Demand': '216',
                                     'Initial Inventory': '1420',
                                     'Product Name': 'Beauty - 300',
                                     'Revenue': '300'}},
                         {'source_row': 3,
                          'values': {'Demand': '263',
                                     'Initial Inventory': '1700',
                                     'Product Name': 'Beauty - 50',
                                     'Revenue': '50'}},
                         {'source_row': 4,
                          'values': {'Demand': '256',
                                     'Initial Inventory': '1690',
                                     'Product Name': 'Beauty - 500',
                                     'Revenue': '500'}},
                         {'source_row': 5,
                          'values': {'Demand': '281',
                                     'Initial Inventory': '1840',
                                     'Product Name': 'Clothing - 25',
                                     'Revenue': '25'}},
                         {'source_row': 6,
                          'values': {'Demand': '261',
                                     'Initial Inventory': '1710',
                                     'Product Name': 'Clothing - 30',
                                     'Revenue': '30'}},
                         {'source_row': 7,
                          'values': {'Demand': '295',
                                     'Initial Inventory': '1930',
                                     'Product Name': 'Clothing - 300',
                                     'Revenue': '300'}},
                         {'source_row': 8,
                          'values': {'Demand': '290',
                                     'Initial Inventory': '1890',
                                     'Product Name': 'Clothing - 50',
                                     'Revenue': '50'}},
                         {'source_row': 9,
                          'values': {'Demand': '244',
                                     'Initial Inventory': '1570',
                                     'Product Name': 'Clothing - 500',
                                     'Revenue': '500'}},
                         {'source_row': 10,
                          'values': {'Demand': '273',
                                     'Initial Inventory': '1810',
                                     'Product Name': 'Electronics - 25',
                                     'Revenue': '25'}},
                         {'source_row': 11,
                          'values': {'Demand': '220',
                                     'Initial Inventory': '1410',
                                     'Product Name': 'Electronics - 30',
                                     'Revenue': '30'}},
                         {'source_row': 12,
                          'values': {'Demand': '286',
                                     'Initial Inventory': '1830',
                                     'Product Name': 'Electronics - 300',
                                     'Revenue': '300'}},
                         {'source_row': 13,
                          'values': {'Demand': '268',
                                     'Initial Inventory': '1750',
                                     'Product Name': 'Electronics - 50',
                                     'Revenue': '50'}},
                         {'source_row': 14,
                          'values': {'Demand': '262',
                                     'Initial Inventory': '1690',
                                     'Product Name': 'Electronics - 500',
                                     'Revenue': '500'}},
                         {'source_row': 15,
                          'values': {'Demand': '255',
                                     'Initial Inventory': '1660',
                                     'Product Name': 'Home Goods - 25',
                                     'Revenue': '25'}},
                         {'source_row': 16,
                          'values': {'Demand': '218',
                                     'Initial Inventory': '1417',
                                     'Product Name': 'Home Goods - 30',
                                     'Revenue': '30'}},
                         {'source_row': 17,
                          'values': {'Demand': '278',
                                     'Initial Inventory': '1807',
                                     'Product Name': 'Home Goods - 50',
                                     'Revenue': '50'}},
                         {'source_row': 18,
                          'values': {'Demand': '195',
                                     'Initial Inventory': '1268',
                                     'Product Name': 'Home Goods - 300',
                                     'Revenue': '300'}},
                         {'source_row': 19,
                          'values': {'Demand': '248',
                                     'Initial Inventory': '1612',
                                     'Product Name': 'Home Goods - 500',
                                     'Revenue': '500'}},
                         {'source_row': 20,
                          'values': {'Demand': '269',
                                     'Initial Inventory': '1749',
                                     'Product Name': 'Sports - 25',
                                     'Revenue': '25'}},
                         {'source_row': 21,
                          'values': {'Demand': '227',
                                     'Initial Inventory': '1476',
                                     'Product Name': 'Sports - 30',
                                     'Revenue': '30'}},
                         {'source_row': 22,
                          'values': {'Demand': '285',
                                     'Initial Inventory': '1853',
                                     'Product Name': 'Sports - 50',
                                     'Revenue': '50'}},
                         {'source_row': 23,
                          'values': {'Demand': '208',
                                     'Initial Inventory': '1352',
                                     'Product Name': 'Sports - 300',
                                     'Revenue': '300'}},
                         {'source_row': 24,
                          'values': {'Demand': '259',
                                     'Initial Inventory': '1684',
                                     'Product Name': 'Sports - 500',
                                     'Revenue': '500'}},
                         {'source_row': 25,
                          'values': {'Demand': '242',
                                     'Initial Inventory': '1573',
                                     'Product Name': 'Furniture - 25',
                                     'Revenue': '25'}},
                         {'source_row': 26,
                          'values': {'Demand': '213',
                                     'Initial Inventory': '1385',
                                     'Product Name': 'Furniture - 30',
                                     'Revenue': '30'}},
                         {'source_row': 27,
                          'values': {'Demand': '272',
                                     'Initial Inventory': '1768',
                                     'Product Name': 'Furniture - 50',
                                     'Revenue': '50'}},
                         {'source_row': 28,
                          'values': {'Demand': '224',
                                     'Initial Inventory': '1456',
                                     'Product Name': 'Furniture - 300',
                                     'Revenue': '300'}},
                         {'source_row': 29,
                          'values': {'Demand': '251',
                                     'Initial Inventory': '1632',
                                     'Product Name': 'Furniture - 500',
                                     'Revenue': '500'}},
                         {'source_row': 30,
                          'values': {'Demand': '257',
                                     'Initial Inventory': '1671',
                                     'Product Name': 'Toys - 25',
                                     'Revenue': '25'}},
                         {'source_row': 31,
                          'values': {'Demand': '235',
                                     'Initial Inventory': '1528',
                                     'Product Name': 'Toys - 30',
                                     'Revenue': '30'}},
                         {'source_row': 32,
                          'values': {'Demand': '291',
                                     'Initial Inventory': '1892',
                                     'Product Name': 'Toys - 50',
                                     'Revenue': '50'}},
                         {'source_row': 33,
                          'values': {'Demand': '199',
                                     'Initial Inventory': '1294',
                                     'Product Name': 'Toys - 300',
                                     'Revenue': '300'}},
                         {'source_row': 34,
                          'values': {'Demand': '264',
                                     'Initial Inventory': '1716',
                                     'Product Name': 'Toys - 500',
                                     'Revenue': '500'}}],
             'returned_rows': 35,
             'role': 'revenue management product-level data',
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
        product = row['Product Name']
        if product in products:
            raise ValueError(f'Duplicate product identifier: {product}')
        products.append(product)
        try:
            revenue[product] = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Invalid Revenue for product {product}: {row['Revenue']}")
        try:
            demand[product] = int(float(row['Demand']))
        except Exception:
            raise ValueError(f"Invalid Demand for product {product}: {row['Demand']}")
        try:
            inventory[product] = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f"Invalid Initial Inventory for product {product}: {row['Initial Inventory']}")
    if not set(revenue) == set(demand) == set(inventory) == set(products):
        raise ValueError('Coefficient coverage mismatch among products.')
    upper_bounds = {i: min(demand[i], inventory[i]) for i in products}
    m = gp.Model('retail_revenue_allocation')
    x_vars = m.addVars(products, lb=0, ub=[upper_bounds[i] for i in products], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
    m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)