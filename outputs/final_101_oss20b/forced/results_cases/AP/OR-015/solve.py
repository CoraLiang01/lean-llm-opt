CSVQA_DATA = {'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RestaurantSalesreport.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1483', 'Initial Inventory': '10440.0', 'Product Name': 'Aalopuri', 'Revenue': '20'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == table_id), None)
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    I = []
    r = {}
    d = {}
    v = {}
    for record in table['records']:
        values = record['values']
        prod = values['Product Name']
        if prod in I:
            raise ValueError(f'Duplicate product name {prod} in data')
        I.append(prod)
        try:
            r[prod] = float(values['Revenue'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Revenue for product {prod}')
        try:
            d[prod] = int(values['Demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Demand for product {prod}')
        try:
            v[prod] = float(values['Initial Inventory'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Initial Inventory for product {prod}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    ub = {i: min(d[i], v[i]) for i in I}
    x = m.addVars(I, lb=0, ub=ub, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
