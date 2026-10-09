CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
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
             'role': 'product revenue, demand, and inventory parameters',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
records = [{'Demand': '765850', 'Initial Inventory': '5627060', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}]
I = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    prod = rec['Product Name']
    I.append(prod)
    try:
        revenue[prod] = float(rec['Revenue'])
        demand[prod] = int(rec['Demand'])
        inventory[prod] = int(rec['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {prod}: {e}')
for prod in I:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f'Missing data for product {prod}')

def build_and_solve():
    model = gp.Model('Baby_Product_Revenue_Maximization')
    model.Params.MIPGap = 0.0001
    x_vars = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x_vars[i] <= inventory[i] for i in I), name='')
    model.addConstrs((x_vars[i] <= demand[i] for i in I), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = build_and_solve()