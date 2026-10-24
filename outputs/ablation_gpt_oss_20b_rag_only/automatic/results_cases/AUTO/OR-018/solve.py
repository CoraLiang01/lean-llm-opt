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
 'tables': [{'columns': ['Revenue', 'Demand', 'Initial Inventory', 'Product Name'],
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'Baby',
                                         'operator': 'contains',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '3066513',
                                     'Initial Inventory': '22749210',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for record in records:
        values = record['values']
        product_name = values['Product Name']
        if 'Baby' not in product_name:
            continue
        I.append(product_name)
        try:
            revenue[product_name] = float(values['Revenue'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid revenue for {product_name}')
        try:
            demand[product_name] = float(values['Demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid demand for {product_name}')
        try:
            inventory[product_name] = float(values['Initial Inventory'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid inventory for {product_name}')
    for i in I:
        if i not in revenue:
            raise ValueError(f'Missing revenue for {i}')
        if i not in demand:
            raise ValueError(f'Missing demand for {i}')
        if i not in inventory:
            raise ValueError(f'Missing inventory for {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    for i in I:
        model.addConstr(x[i] <= inventory[i])
        model.addConstr(x[i] <= demand[i])
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()