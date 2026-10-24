CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘27in’ product i that the company plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SalesDataAnalysis.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '149.99'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = None
    for t in data.get('tables', []):
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Required table file_0_view_0 not found')
    records = table.get('records', [])
    I = []
    r = {}
    d = {}
    s = {}
    for rec in records:
        vals = rec.get('values', {})
        prod = vals.get('Product Name')
        if prod is None:
            raise ValueError('Missing Product Name in record')
        if prod in I:
            raise ValueError(f'Duplicate product name {prod}')
        I.append(prod)
        rev_str = vals.get('Revenue')
        if rev_str is None:
            raise ValueError(f'Missing Revenue for {prod}')
        try:
            rev = float(rev_str)
        except Exception:
            raise ValueError(f'Invalid Revenue value for {prod}')
        r[prod] = rev
        dem_str = vals.get('Demand')
        if dem_str is None:
            raise ValueError(f'Missing Demand for {prod}')
        try:
            dem = float(dem_str)
        except Exception:
            raise ValueError(f'Invalid Demand value for {prod}')
        d[prod] = dem
        inv_str = vals.get('Initial Inventory')
        if inv_str is None:
            raise ValueError(f'Missing Initial Inventory for {prod}')
        try:
            inv = float(inv_str)
        except Exception:
            raise ValueError(f'Invalid Initial Inventory value for {prod}')
        s[prod] = inv
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.addConstrs((x[i] <= s[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
