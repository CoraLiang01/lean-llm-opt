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
                                         'evidence': 'products classified under ‘27in’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
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
records = [{'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Demand': '12474', 'Initial Inventory': '62440'}, {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Demand': '15057', 'Initial Inventory': '75500'}]
product_indices = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    name = rec['Product Name']
    product_indices.append(name)
    try:
        revenue[name] = float(rec['Revenue'])
        demand[name] = int(rec['Demand'])
        inventory[name] = int(rec['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {name}: {e}')
for i in product_indices:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for product {i}')

def build_model():
    m = gp.Model('27in_Product_Revenue_Max')
    x_vars = m.addVars(product_indices, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_indices)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= inventory[i] for i in product_indices), name='')
    m.addConstrs((x_vars[i] <= demand[i] for i in product_indices), name='')
    m.Params.MIPGap = 0.0001
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')