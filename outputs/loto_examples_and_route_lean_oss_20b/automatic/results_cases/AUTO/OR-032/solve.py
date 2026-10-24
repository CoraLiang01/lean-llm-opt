CSVQA_DATA = {'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'DifferentStoreSales.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1980', 'Initial Inventory': '9920.0', 'Product_Name': 'Books_15.15', 'Revenue': '15.15'}}, {'data_row_number': 1, 'values': {'Demand': '3024', 'Initial Inventory': '20160.0', 'Product_Name': 'Books_30.3', 'Revenue': '30.3'}}, {'data_row_number': 2, 'values': {'Demand': '4536', 'Initial Inventory': '30000.0', 'Product_Name': 'Books_45.45', 'Revenue': '45.45'}}, {'data_row_number': 3, 'values': {'Demand': '5601', 'Initial Inventory': '38360.0', 'Product_Name': 'Books_60.6', 'Revenue': '60.6'}}, {'data_row_number': 4, 'values': {'Demand': '7567', 'Initial Inventory': '51450.0', 'Product_Name': 'Books_75.75', 'Revenue': '75.75'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    table_id = 'file_0_view_0'
    table = None
    for t in CSVQA_DATA['tables']:
        if t.get('table_id') == table_id:
            table = t
            break
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    records = table.get('records', [])
    if not records:
        raise ValueError(f'No records found in table {table_id}')
    rev = {}
    dem = {}
    inv = {}
    data_row_numbers = []
    for rec in records:
        sr = rec.get('data_row_number')
        if sr is None:
            raise ValueError('Record missing data_row_number')
        vals = rec.get('values', {})
        for col in ('Revenue', 'Demand', 'Initial Inventory'):
            if col not in vals:
                raise ValueError(f"Missing column '{col}' in record {sr}")
        try:
            rev_val = float(vals['Revenue'])
            dem_val = int(vals['Demand'])
            inv_val = float(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {sr}: {e}')
        rev[sr] = rev_val
        dem[sr] = dem_val
        inv[sr] = inv_val
        data_row_numbers.append(sr)
    x = model.addVars(data_row_numbers, lb=0, vtype=GRB.INTEGER, name='')
    for i in data_row_numbers:
        model.addConstr(x[i] <= dem[i], name='')
        model.addConstr(x[i] <= inv[i], name='')
    model.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in data_row_numbers)), GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
