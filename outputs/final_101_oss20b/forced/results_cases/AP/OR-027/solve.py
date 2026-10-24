CSVQA_DATA = {'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘Organ’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv', 'records': [{'data_row_number': 17, 'values': {'Demand': '678906', 'Initial Inventory': '5034020.0', 'Revenue': '60.8', 'Sub Category': 'Organic Fruits'}}, {'data_row_number': 18, 'values': {'Demand': '749927', 'Initial Inventory': '5589290.0', 'Revenue': '918.45', 'Sub Category': 'Organic Staples'}}, {'data_row_number': 19, 'values': {'Demand': '699808', 'Initial Inventory': '5202710.0', 'Revenue': '77.52', 'Sub Category': 'Organic Vegetables'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = None
    for t in tables:
        if t.get('table_id') == table_id:
            table = t
            break
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    records = table.get('records', [])
    col_map = {'product_id': 'Sub Category', 'rev_i': 'Revenue', 'dem_i': 'Demand', 'inv_i': 'Initial Inventory'}
    I = []
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        data_row_number = rec.get('data_row_number')
        if data_row_number is None:
            raise ValueError('Record missing data_row_number')
        values = rec.get('values', {})
        for key, col in col_map.items():
            if col not in values:
                raise ValueError(f'Missing column {col} in record {data_row_number}')
        try:
            rev_val = float(values[col_map['rev_i']])
            dem_val = float(values[col_map['dem_i']])
            inv_val = float(values[col_map['inv_i']])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {data_row_number}: {e}')
        I.append(data_row_number)
        rev[data_row_number] = rev_val
        dem[data_row_number] = dem_val
        inv[data_row_number] = inv_val
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= dem[i])
        m.addConstr(x[i] <= inv[i])
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
