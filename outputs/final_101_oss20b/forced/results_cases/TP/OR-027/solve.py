CSVQA_DATA = {'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘Organ’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv', 'records': [{'data_row_number': 17, 'values': {'Demand': '678906', 'Initial Inventory': '5034020.0', 'Revenue': '60.8', 'Sub Category': 'Organic Fruits'}}, {'data_row_number': 18, 'values': {'Demand': '749927', 'Initial Inventory': '5589290.0', 'Revenue': '918.45', 'Sub Category': 'Organic Staples'}}, {'data_row_number': 19, 'values': {'Demand': '699808', 'Initial Inventory': '5202710.0', 'Revenue': '77.52', 'Sub Category': 'Organic Vegetables'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Required table file_0_view_0 not found in CSVQA_DATA')
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    I = set()
    r = {}
    d = {}
    s = {}
    for rec in records:
        values = rec.get('values', {})
        subcat = values.get('Sub Category')
        if subcat is None:
            raise ValueError('Missing Sub Category in a record')
        I.add(subcat)
        try:
            rev = float(values.get('Revenue'))
        except (TypeError, ValueError):
            raise ValueError(f'Invalid Revenue for {subcat}')
        try:
            dem = float(values.get('Demand'))
        except (TypeError, ValueError):
            raise ValueError(f'Invalid Demand for {subcat}')
        try:
            inv = float(values.get('Initial Inventory'))
        except (TypeError, ValueError):
            raise ValueError(f'Invalid Initial Inventory for {subcat}')
        r[subcat] = rev
        d[subcat] = dem
        s[subcat] = inv
    for i in I:
        if i not in r or i not in d or i not in s:
            raise ValueError(f'Missing parameter for {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= s[i], name=f"supply_{i.replace(' ', '_')}")
        m.addConstr(x[i] <= d[i], name=f"demand_{i.replace(' ', '_')}")
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
