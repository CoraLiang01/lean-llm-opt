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
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}},
                         {'source_row': 1,
                          'values': {'Demand': '2961484',
                                     'Initial Inventory': '22049510',
                                     'Product Name': 'Beverages_47.45',
                                     'Revenue': '47.45'}},
                         {'source_row': 2,
                          'values': {'Demand': '2621950',
                                     'Initial Inventory': '19459680',
                                     'Product Name': 'Cereal_205.7',
                                     'Revenue': '205.7'}},
                         {'source_row': 3,
                          'values': {'Demand': '2660974',
                                     'Initial Inventory': '19754410',
                                     'Product Name': 'Clothes_109.28',
                                     'Revenue': '109.28'}},
                         {'source_row': 4,
                          'values': {'Demand': '2896197',
                                     'Initial Inventory': '21366410',
                                     'Product Name': 'Cosmetics_437.2',
                                     'Revenue': '437.2'}},
                         {'source_row': 5,
                          'values': {'Demand': '3169426',
                                     'Initial Inventory': '23410830',
                                     'Product Name': 'Fruits_9.33',
                                     'Revenue': '9.33'}},
                         {'source_row': 6,
                          'values': {'Demand': '2846953',
                                     'Initial Inventory': '20986130',
                                     'Product Name': 'Household_668.27',
                                     'Revenue': '668.27'}},
                         {'source_row': 7,
                          'values': {'Demand': '2546972',
                                     'Initial Inventory': '19011970',
                                     'Product Name': 'Meat_421.89',
                                     'Revenue': '421.89'}},
                         {'source_row': 8,
                          'values': {'Demand': '2855686',
                                     'Initial Inventory': '21062780',
                                     'Product Name': 'Office Supplies_651.21',
                                     'Revenue': '651.21'}},
                         {'source_row': 9,
                          'values': {'Demand': '2855360',
                                     'Initial Inventory': '21265920',
                                     'Product Name': 'Personal Care_81.73',
                                     'Revenue': '81.73'}},
                         {'source_row': 10,
                          'values': {'Demand': '2592261',
                                     'Initial Inventory': '19155280',
                                     'Product Name': 'Snacks_152.58',
                                     'Revenue': '152.58'}},
                         {'source_row': 11,
                          'values': {'Demand': '2826603',
                                     'Initial Inventory': '20867490',
                                     'Product Name': 'Vegetables_154.06',
                                     'Revenue': '154.06'}}],
             'returned_rows': 12,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
    records = table['records']
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        pname = rec['values']['Product Name']
        if pname.startswith('Baby'):
            items.append(pname)
            try:
                revenue[pname] = float(rec['values']['Revenue'])
                demand[pname] = int(rec['values']['Demand'])
                inventory[pname] = int(rec['values']['Initial Inventory'])
            except Exception as e:
                raise ValueError(f"Invalid data for product '{pname}': {e}")
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f"Missing data for product '{i}'.")
    m = gp.Model('Baby_Product_Revenue_Max')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()