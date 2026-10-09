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
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
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
        vals = rec['values']
        prod = vals['Product Name']
        items.append(prod)
        try:
            revenue[prod] = float(vals['Revenue'])
            demand[prod] = int(vals['Demand'])
            inventory[prod] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Invalid data for product '{prod}': {e}")
    for prod in items:
        if prod not in revenue or prod not in demand or prod not in inventory:
            raise ValueError(f"Missing data for product '{prod}'.")
    m = gp.Model('Baby_Product_Revenue_Max')
    x = m.addVars(items, lb=0, ub={i: min(demand[i], inventory[i]) for i in items}, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')