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
                                         'evidence': '‘Baby’ product',
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
             'role': 'revenue management data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = [rec['values'] for rec in CSVQA_DATA['tables'][0]['records']]
if not table:
    raise ValueError("No records found for 'Baby' products in file_0_view_0.")
I = []
revenue = {}
demand = {}
inventory = {}
for rec in table:
    product = rec['Product Name']
    try:
        A_i = float(rec['Revenue'])
        d_i = int(rec['Demand'])
        I_i = int(rec['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {product}: {e}')
    I.append(product)
    revenue[product] = A_i
    demand[product] = d_i
    inventory[product] = I_i
if not set(revenue) == set(demand) == set(inventory) == set(I):
    raise ValueError('Mismatch in index sets for parameters.')
m = gp.Model('Baby_Product_Revenue_Maximization')
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x_vars[i] <= min(inventory[i], demand[i]), name=f'ub_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'lb_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')