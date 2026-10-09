CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'DifferentStoreSales.csv',
             'filters': {'conditions': [{'column': 'Product_Name',
                                         'dtype': 'string',
                                         'evidence': '‘Books’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Books_'}],
                         'logic': 'and'},
             'original_rows': 40,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1980',
                                     'Initial Inventory': '9920.0',
                                     'Product_Name': 'Books_15.15',
                                     'Revenue': '15.15'}},
                         {'source_row': 1,
                          'values': {'Demand': '3024',
                                     'Initial Inventory': '20160.0',
                                     'Product_Name': 'Books_30.3',
                                     'Revenue': '30.3'}},
                         {'source_row': 2,
                          'values': {'Demand': '4536',
                                     'Initial Inventory': '30000.0',
                                     'Product_Name': 'Books_45.45',
                                     'Revenue': '45.45'}},
                         {'source_row': 3,
                          'values': {'Demand': '5601',
                                     'Initial Inventory': '38360.0',
                                     'Product_Name': 'Books_60.6',
                                     'Revenue': '60.6'}},
                         {'source_row': 4,
                          'values': {'Demand': '7567',
                                     'Initial Inventory': '51450.0',
                                     'Product_Name': 'Books_75.75',
                                     'Revenue': '75.75'}}],
             'returned_rows': 5,
             'role': 'product revenue and constraints',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = [{'Demand': '1980', 'Initial Inventory': '9920.0', 'Product_Name': 'Books_15.15', 'Revenue': '15.15'}, {'Demand': '3024', 'Initial Inventory': '20160.0', 'Product_Name': 'Books_30.3', 'Revenue': '30.3'}, {'Demand': '4536', 'Initial Inventory': '30000.0', 'Product_Name': 'Books_45.45', 'Revenue': '45.45'}, {'Demand': '5601', 'Initial Inventory': '38360.0', 'Product_Name': 'Books_60.6', 'Revenue': '60.6'}, {'Demand': '7567', 'Initial Inventory': '51450.0', 'Product_Name': 'Books_75.75', 'Revenue': '75.75'}]
products = []
revenue = {}
inventory = {}
demand = {}
for row in table:
    i = row['Product_Name']
    products.append(i)
    try:
        revenue[i] = float(row['Revenue'])
        inventory[i] = float(row['Initial Inventory'])
        demand[i] = int(row['Demand'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {i}: {e}')
if not set(revenue) == set(inventory) == set(demand) == set(products):
    raise ValueError('Mismatch in product indices among revenue, inventory, and demand.')

def build_and_solve():
    m = gp.Model('Books_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
    m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve()