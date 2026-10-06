CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'Baby' product i",
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product decision and parameter table',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA['tables'][0]['records']
    products = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in data:
        pname = rec['values']['Product Name']
        if pname.startswith('Baby'):
            products.append(pname)
            try:
                revenue[pname] = float(rec['values']['Revenue'])
                demand[pname] = int(rec['values']['Demand'])
                inventory[pname] = int(rec['values']['Initial Inventory'])
            except Exception as e:
                raise ValueError(f'Invalid data for product {pname}: {e}')
    for pname in products:
        if pname not in revenue or pname not in demand or pname not in inventory:
            raise ValueError(f'Missing data for product {pname}')
    m = gp.Model('Baby_Product_Revenue_Maximization')
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in products), name='')
    m.addConstrs((x[i] <= inventory[i] for i in products), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()