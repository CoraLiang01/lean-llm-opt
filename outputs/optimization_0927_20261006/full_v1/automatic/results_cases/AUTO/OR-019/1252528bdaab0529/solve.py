CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'27in' products",
                                         'operator': 'prefix',
                                         'value': '27in'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'27in' products",
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'or'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}}],
             'returned_rows': 2,
             'role': 'product revenue, demand, and inventory for 27in products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = CSVQA_DATA['tables'][0]
records = table['records']
I = []
A = {}
d = {}
I_param = {}
for rec in records:
    pname = rec['values']['Product Name']
    revenue = rec['values']['Revenue']
    demand = rec['values']['Demand']
    inventory = rec['values']['Initial Inventory']
    I.append(pname)
    try:
        A[pname] = float(revenue)
    except Exception:
        raise ValueError(f"Revenue missing or invalid for product '{pname}'")
    try:
        d[pname] = int(demand)
    except Exception:
        raise ValueError(f"Demand missing or invalid for product '{pname}'")
    try:
        I_param[pname] = int(inventory)
    except Exception:
        raise ValueError(f"Initial Inventory missing or invalid for product '{pname}'")
if not set(I) == set(A) == set(d) == set(I_param):
    raise ValueError('Mismatch in index sets for products, revenue, demand, or inventory.')
m = gp.Model('27in_Product_Revenue_Optimization')
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
for i in I:
    ub = min(I_param[i], d[i])
    m.addConstr(x_vars[i] <= ub, name=f'ub_{i}')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')