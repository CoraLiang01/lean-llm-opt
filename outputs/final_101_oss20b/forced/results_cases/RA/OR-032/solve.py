CSVQA_DATA = {'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'DifferentStoreSales.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1980', 'Initial Inventory': '9920.0', 'Product_Name': 'Books_15.15', 'Revenue': '15.15'}}, {'data_row_number': 1, 'values': {'Demand': '3024', 'Initial Inventory': '20160.0', 'Product_Name': 'Books_30.3', 'Revenue': '30.3'}}, {'data_row_number': 2, 'values': {'Demand': '4536', 'Initial Inventory': '30000.0', 'Product_Name': 'Books_45.45', 'Revenue': '45.45'}}, {'data_row_number': 3, 'values': {'Demand': '5601', 'Initial Inventory': '38360.0', 'Product_Name': 'Books_60.6', 'Revenue': '60.6'}}, {'data_row_number': 4, 'values': {'Demand': '7567', 'Initial Inventory': '51450.0', 'Product_Name': 'Books_75.75', 'Revenue': '75.75'}}], 'table_id': 'file_0_view_0'}]}
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
        raise ValueError('Required table file_0_view_0 not found')
    required_cols = ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in table.get('columns', []):
            raise ValueError(f'Missing required column {col}')
    records = table.get('records', [])
    rev = {}
    dem = {}
    inv = {}
    products = []
    for rec in records:
        vals = rec.get('values', {})
        prod = vals.get('Product_Name')
        if prod is None:
            raise ValueError('Missing Product_Name in record')
        try:
            rev_val = float(vals.get('Revenue', '0'))
        except Exception:
            raise ValueError(f'Invalid Revenue for product {prod}')
        try:
            dem_val = int(float(vals.get('Demand', '0')))
        except Exception:
            raise ValueError(f'Invalid Demand for product {prod}')
        try:
            inv_val = float(vals.get('Initial Inventory', '0'))
        except Exception:
            raise ValueError(f'Invalid Initial Inventory for product {prod}')
        rev[prod] = rev_val
        dem[prod] = dem_val
        inv[prod] = inv_val
        products.append(prod)
    for prod in products:
        if prod not in rev or prod not in dem or prod not in inv:
            raise ValueError(f'Missing parameter for product {prod}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((rev[p] * x[p] for p in products)), GRB.MAXIMIZE)
    model.addConstrs((x[p] <= dem[p] for p in products), name='')
    model.addConstrs((x[p] <= inv[p] for p in products), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
