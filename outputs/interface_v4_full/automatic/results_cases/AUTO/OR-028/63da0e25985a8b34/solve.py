CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A company offers a variety of products with revenue data provided in the ‘Revenue’ column. The goal is to '
          'determine which demands to meet within the sales horizon using the initial inventory specified in the '
          '‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be '
          'deterministic and known in advance. Decision variables x_i represent the number of units of each product '
          'that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'WomenClothingEcommerceSalesData.csv',
             'filters': {},
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    table = [{'Product Name': 'sku_I27', 'Revenue': 238, 'Demand': 6, 'Initial Inventory': 30}, {'Product Name': 'sku_I499', 'Revenue': 287, 'Demand': 4, 'Initial Inventory': 20}, {'Product Name': 'sku_I719', 'Revenue': 268, 'Demand': 16, 'Initial Inventory': 80}, {'Product Name': 'sku_T18', 'Revenue': 318, 'Demand': 14, 'Initial Inventory': 70}, {'Product Name': 'sku_T29', 'Revenue': 207, 'Demand': 4, 'Initial Inventory': 20}, {'Product Name': 'sku_T39', 'Revenue': 258, 'Demand': 32, 'Initial Inventory': 160}, {'Product Name': 'sku_T499', 'Revenue': 249, 'Demand': 8, 'Initial Inventory': 40}, {'Product Name': 'sku_T9', 'Revenue': 227, 'Demand': 2, 'Initial Inventory': 10}, {'Product Name': 'sku_3081', 'Revenue': 198, 'Demand': 10, 'Initial Inventory': 50}, {'Product Name': 'sku_339', 'Revenue': 254, 'Demand': 8, 'Initial Inventory': 40}, {'Product Name': 'sku_3799', 'Revenue': 246, 'Demand': 18, 'Initial Inventory': 90}, {'Product Name': 'sku_439', 'Revenue': 258, 'Demand': 2, 'Initial Inventory': 10}, {'Product Name': 'sku_539', 'Revenue': 268, 'Demand': 4, 'Initial Inventory': 20}, {'Product Name': 'sku_61399', 'Revenue': 278, 'Demand': 8, 'Initial Inventory': 40}, {'Product Name': 'sku_628', 'Revenue': 268, 'Demand': 2, 'Initial Inventory': 10}, {'Product Name': 'sku_708', 'Revenue': 298, 'Demand': 198, 'Initial Inventory': 990}, {'Product Name': 'sku_77', 'Revenue': 258, 'Demand': 32, 'Initial Inventory': 160}, {'Product Name': 'sku_79', 'Revenue': 315, 'Demand': 18, 'Initial Inventory': 90}, {'Product Name': 'sku_799', 'Revenue': 264, 'Demand': 570, 'Initial Inventory': 2870}, {'Product Name': 'sku_8499', 'Revenue': 238, 'Demand': 6, 'Initial Inventory': 30}, {'Product Name': 'sku_89', 'Revenue': 258, 'Demand': 26, 'Initial Inventory': 130}, {'Product Name': 'sku_897', 'Revenue': 268, 'Demand': 6, 'Initial Inventory': 30}, {'Product Name': 'sku_9699', 'Revenue': 288, 'Demand': 33, 'Initial Inventory': 170}, {'Product Name': 'sku_bobo', 'Revenue': 228, 'Demand': 33, 'Initial Inventory': 170}]
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in table:
        i = rec['Product Name']
        items.append(i)
        try:
            revenue[i] = int(rec['Revenue'])
            demand[i] = int(rec['Demand'])
            inventory[i] = int(rec['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {i}: {e}')
    if not set(revenue) == set(demand) == set(inventory) == set(items):
        raise ValueError('Mismatch in index sets for parameters.')
    m = gp.Model('WomenClothingEcommerceSales')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()