CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Revenue', 'Demand', 'Initial Inventory', 'Product Name'], 'file_name': 'Salesdata.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '3066513', 'Initial Inventory': '22749210', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError('Required table file_0_view_0 not found in CSVQA_DATA')
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for record in table['records']:
        values = record['values']
        prod_name = values['Product Name']
        if 'Baby' not in prod_name:
            continue
        I.append(prod_name)
        try:
            revenue[prod_name] = float(values['Revenue'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Revenue for product {prod_name}')
        try:
            demand[prod_name] = int(values['Demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Demand for product {prod_name}')
        try:
            inventory[prod_name] = int(values['Initial Inventory'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Initial Inventory for product {prod_name}')
    for prod in I:
        if prod not in revenue or prod not in demand or prod not in inventory:
            raise ValueError(f'Incomplete data for product {prod}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= demand[i] for i in I), name='')
    model.addConstrs((x[i] <= inventory[i] for i in I), name='')
    model.optimize()
    return model
m = solve_problem()
