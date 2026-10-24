CSVQA_DATA = {'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ column. The dealership aims to maximize total revenue using the initial inventory of car models classified under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'BigMartSales.csv', 'records': [{'data_row_number': 1163, 'values': {'Demand': '30', 'Initial Inventory': '200', 'Product Name': 'FDK57', 'Revenue': '119.144'}}, {'data_row_number': 1501, 'values': {'Demand': '40', 'Initial Inventory': '100', 'Product Name': 'FDK57', 'Revenue': '119.144'}}, {'data_row_number': 1576, 'values': {'Demand': '30', 'Initial Inventory': '200', 'Product Name': 'FDK57', 'Revenue': '121.244'}}, {'data_row_number': 1793, 'values': {'Demand': '50', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.144'}}, {'data_row_number': 2438, 'values': {'Demand': '10', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.544'}}, {'data_row_number': 4297, 'values': {'Demand': '30', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '121.244'}}, {'data_row_number': 4806, 'values': {'Demand': '50', 'Initial Inventory': '250', 'Product Name': 'FDK57', 'Revenue': '119.744'}}, {'data_row_number': 4942, 'values': {'Demand': '50', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.844'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'))
    records = table['records']
    I = []
    r = {}
    d = {}
    v = {}
    for rec in records:
        sid = rec['data_row_number']
        if sid in I:
            raise ValueError(f'Duplicate data_row_number {sid}')
        I.append(sid)
        vals = rec['values']
        try:
            rev = float(vals['Revenue'])
            dem = int(vals['Demand'])
            inv = int(vals['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing parameter {e} in record {sid}')
        r[sid] = rev
        d[sid] = dem
        v[sid] = inv
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i], name=f'dem_{i}')
        m.addConstr(x[i] <= v[i], name=f'inv_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
