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
                                         'evidence': 'products classified under ‘Baby’',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = [rec['values'] for rec in CSVQA_DATA['tables'][0]['records']]
if not table:
    raise ValueError("No records found for products classified under 'Baby'.")
I = []
A = {}
d = {}
I_inv = {}
for row in table:
    product = row['Product Name']
    try:
        revenue = float(row['Revenue'])
        demand = int(row['Demand'])
        initial_inventory = int(row['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {product}: {e}')
    I.append(product)
    A[product] = revenue
    d[product] = demand
    I_inv[product] = initial_inventory
if not set(A) == set(d) == set(I_inv) == set(I):
    raise ValueError('Parameter keys do not match index set I.')
m = gp.Model('Original_RAG_NRM')
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I_inv[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')