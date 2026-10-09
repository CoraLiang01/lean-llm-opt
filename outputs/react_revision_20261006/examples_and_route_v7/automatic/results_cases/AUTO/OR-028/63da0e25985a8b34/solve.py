CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A company offers a variety of products with revenue data provided in the ‘Revenue’ column. The goal is to '
          'determine which demands to meet within the sales horizon using the initial inventory specified in the '
          '‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be '
          'deterministic and known in advance. Decision variables x_i represent the number of units of each product '
          'that will be fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'WomenClothingEcommerceSalesData.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 24,
             'records': [{'source_row': 0,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'sku_I27',
                                     'Revenue': '238'}},
                         {'source_row': 1,
                          'values': {'Demand': '4',
                                     'Initial Inventory': '20',
                                     'Product Name': 'sku_I499',
                                     'Revenue': '287'}},
                         {'source_row': 2,
                          'values': {'Demand': '16',
                                     'Initial Inventory': '80',
                                     'Product Name': 'sku_I719',
                                     'Revenue': '268'}},
                         {'source_row': 3,
                          'values': {'Demand': '14',
                                     'Initial Inventory': '70',
                                     'Product Name': 'sku_T18',
                                     'Revenue': '318'}},
                         {'source_row': 4,
                          'values': {'Demand': '4',
                                     'Initial Inventory': '20',
                                     'Product Name': 'sku_T29',
                                     'Revenue': '207'}},
                         {'source_row': 5,
                          'values': {'Demand': '32',
                                     'Initial Inventory': '160',
                                     'Product Name': 'sku_T39',
                                     'Revenue': '258'}},
                         {'source_row': 6,
                          'values': {'Demand': '8',
                                     'Initial Inventory': '40',
                                     'Product Name': 'sku_T499',
                                     'Revenue': '249'}},
                         {'source_row': 7,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'sku_T9',
                                     'Revenue': '227'}},
                         {'source_row': 8,
                          'values': {'Demand': '10',
                                     'Initial Inventory': '50',
                                     'Product Name': 'sku_3081',
                                     'Revenue': '198'}},
                         {'source_row': 9,
                          'values': {'Demand': '8',
                                     'Initial Inventory': '40',
                                     'Product Name': 'sku_339',
                                     'Revenue': '254'}},
                         {'source_row': 10,
                          'values': {'Demand': '18',
                                     'Initial Inventory': '90',
                                     'Product Name': 'sku_3799',
                                     'Revenue': '246'}},
                         {'source_row': 11,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'sku_439',
                                     'Revenue': '258'}},
                         {'source_row': 12,
                          'values': {'Demand': '4',
                                     'Initial Inventory': '20',
                                     'Product Name': 'sku_539',
                                     'Revenue': '268'}},
                         {'source_row': 13,
                          'values': {'Demand': '8',
                                     'Initial Inventory': '40',
                                     'Product Name': 'sku_61399',
                                     'Revenue': '278'}},
                         {'source_row': 14,
                          'values': {'Demand': '2',
                                     'Initial Inventory': '10',
                                     'Product Name': 'sku_628',
                                     'Revenue': '268'}},
                         {'source_row': 15,
                          'values': {'Demand': '198',
                                     'Initial Inventory': '990',
                                     'Product Name': 'sku_708',
                                     'Revenue': '298'}},
                         {'source_row': 16,
                          'values': {'Demand': '32',
                                     'Initial Inventory': '160',
                                     'Product Name': 'sku_77',
                                     'Revenue': '258'}},
                         {'source_row': 17,
                          'values': {'Demand': '18',
                                     'Initial Inventory': '90',
                                     'Product Name': 'sku_79',
                                     'Revenue': '315'}},
                         {'source_row': 18,
                          'values': {'Demand': '570',
                                     'Initial Inventory': '2870',
                                     'Product Name': 'sku_799',
                                     'Revenue': '264'}},
                         {'source_row': 19,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'sku_8499',
                                     'Revenue': '238'}},
                         {'source_row': 20,
                          'values': {'Demand': '26',
                                     'Initial Inventory': '130',
                                     'Product Name': 'sku_89',
                                     'Revenue': '258'}},
                         {'source_row': 21,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '30',
                                     'Product Name': 'sku_897',
                                     'Revenue': '268'}},
                         {'source_row': 22,
                          'values': {'Demand': '33',
                                     'Initial Inventory': '170',
                                     'Product Name': 'sku_9699',
                                     'Revenue': '288'}},
                         {'source_row': 23,
                          'values': {'Demand': '33',
                                     'Initial Inventory': '170',
                                     'Product Name': 'sku_bobo',
                                     'Revenue': '228'}}],
             'returned_rows': 24,
             'role': 'product demand and revenue data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    df = CSVQA_FRAMES['file_0_view_0']
    products = df['Product Name'].tolist()
    try:
        revenue = df.set_index('Product Name')['Revenue'].astype(float).to_dict()
        demand = df.set_index('Product Name')['Demand'].astype(float).to_dict()
        inventory = df.set_index('Product Name')['Initial Inventory'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting parameters: {e}')
    for i in products:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product {i}')
    m = gp.Model('product_fulfillment')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in products), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in products), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()