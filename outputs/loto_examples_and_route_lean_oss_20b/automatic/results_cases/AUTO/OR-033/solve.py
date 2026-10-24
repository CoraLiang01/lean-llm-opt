CSVQA_DATA = {'query': 'The company operates in the European market and offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'EuropeSalesRecords.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '765850', 'Initial Inventory': '5627060', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    I = [rec['data_row_number'] for rec in records]
    revenue = {}
    demand = {}
    initinv = {}
    for rec in records:
        src = rec['data_row_number']
        vals = rec['values']
        for key in ('Revenue', 'Demand', 'Initial Inventory'):
            if key not in vals:
                raise ValueError(f"Missing parameter '{key}' for data_row_number {src}")
        try:
            rev = float(vals['Revenue'])
        except Exception:
            raise ValueError(f"Invalid Revenue value for data_row_number {src}: {vals['Revenue']}")
        try:
            dem = int(vals['Demand'])
        except Exception:
            raise ValueError(f"Invalid Demand value for data_row_number {src}: {vals['Demand']}")
        try:
            inv = int(vals['Initial Inventory'])
        except Exception:
            raise ValueError(f"Invalid Initial Inventory value for data_row_number {src}: {vals['Initial Inventory']}")
        revenue[src] = rev
        demand[src] = dem
        initinv[src] = inv
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in I), name='')
    m.addConstrs((x[i] <= initinv[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
