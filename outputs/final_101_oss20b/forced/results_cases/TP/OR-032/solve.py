CSVQA_DATA = {'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'DifferentStoreSales.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1980', 'Initial Inventory': '9920.0', 'Product_Name': 'Books_15.15', 'Revenue': '15.15'}}, {'data_row_number': 1, 'values': {'Demand': '3024', 'Initial Inventory': '20160.0', 'Product_Name': 'Books_30.3', 'Revenue': '30.3'}}, {'data_row_number': 2, 'values': {'Demand': '4536', 'Initial Inventory': '30000.0', 'Product_Name': 'Books_45.45', 'Revenue': '45.45'}}, {'data_row_number': 3, 'values': {'Demand': '5601', 'Initial Inventory': '38360.0', 'Product_Name': 'Books_60.6', 'Revenue': '60.6'}}, {'data_row_number': 4, 'values': {'Demand': '7567', 'Initial Inventory': '51450.0', 'Product_Name': 'Books_75.75', 'Revenue': '75.75'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    r = {}
    d = {}
    s = {}
    for rec in records:
        values = rec.get('values', {})
        prod = values.get('Product_Name')
        if prod is None:
            raise ValueError(f"Missing Product_Name in record {rec.get('data_row_number')}")
        if prod in r:
            raise ValueError(f'Duplicate product name detected: {prod}')
        try:
            rev = float(values['Revenue'])
        except (KeyError, ValueError):
            raise ValueError(f'Invalid or missing Revenue for product {prod}')
        try:
            dem = int(values['Demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Invalid or missing Demand for product {prod}')
        try:
            inv = float(values['Initial Inventory'])
        except (KeyError, ValueError):
            raise ValueError(f'Invalid or missing Initial Inventory for product {prod}')
        r[prod] = rev
        d[prod] = dem
        s[prod] = inv
    I = list(r.keys())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i])
        m.addConstr(x[i] <= s[i])
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
