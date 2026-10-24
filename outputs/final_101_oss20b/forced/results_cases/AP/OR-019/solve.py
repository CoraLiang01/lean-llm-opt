CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘27in’ product i that the company plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SalesDataAnalysis.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '149.99'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    I = []
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        data_row_number = rec.get('data_row_number')
        if data_row_number is None:
            raise ValueError("Record missing 'data_row_number' key")
        values = rec.get('values', {})
        if 'Revenue' not in values:
            raise ValueError(f"Missing 'Revenue' for data_row_number {data_row_number}")
        if 'Demand' not in values:
            raise ValueError(f"Missing 'Demand' for data_row_number {data_row_number}")
        if 'Initial Inventory' not in values:
            raise ValueError(f"Missing 'Initial Inventory' for data_row_number {data_row_number}")
        try:
            rev_val = float(values['Revenue'])
            dem_val = int(values['Demand'])
            inv_val = int(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {data_row_number}: {e}')
        I.append(data_row_number)
        rev[data_row_number] = rev_val
        dem[data_row_number] = dem_val
        inv[data_row_number] = inv_val
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    for i in I:
        model.addConstr(x[i] <= dem[i], name=f'dem_{i}')
        model.addConstr(x[i] <= inv[i], name=f'inv_{i}')
    model.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for variable in model.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
