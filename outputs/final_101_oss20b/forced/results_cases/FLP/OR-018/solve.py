CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesdata.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '3066513', 'Initial Inventory': '22749210', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
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
    records = table.get('records', [])
    I = []
    r = {}
    d = {}
    inv = {}
    for rec in records:
        vals = rec['values']
        prod_name = vals.get('Product Name', '').strip()
        if 'Baby' not in prod_name:
            continue
        I.append(prod_name)
        try:
            r[prod_name] = float(vals.get('Revenue', 0))
            d[prod_name] = float(vals.get('Demand', 0))
            inv[prod_name] = float(vals.get('Initial Inventory', 0))
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid numeric value for product {prod_name}: {e}')
    for i in I:
        if i not in r or i not in d or i not in inv:
            raise ValueError(f'Missing parameter for product {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inv[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
