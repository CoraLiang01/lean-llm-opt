CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesdata.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '3066513', 'Initial Inventory': '22749210', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    if table is None:
        raise ValueError('Required table file_0_view_0 not found in CSVQA_DATA')
    records = table.get('records', [])
    if not records:
        raise ValueError('No records returned for Baby products')
    I = [rec['data_row_number'] for rec in records]
    Revenue = {}
    Demand = {}
    Inv = {}
    for rec in records:
        sid = rec['data_row_number']
        vals = rec.get('values', {})
        for field in ('Revenue', 'Demand', 'Initial Inventory'):
            if field not in vals:
                raise ValueError(f"Missing field '{field}' in record {sid}")
        try:
            rev = float(vals['Revenue'])
            dem = int(vals['Demand'])
            inv = int(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {sid}: {e}')
        Revenue[sid] = rev
        Demand[sid] = dem
        Inv[sid] = inv
    for i in I:
        if i not in Revenue or i not in Demand or i not in Inv:
            raise ValueError(f'Missing parameter for index {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((Revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= Inv[i], name=f'Inv_{i}')
        m.addConstr(x[i] <= Demand[i], name=f'Demand_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
