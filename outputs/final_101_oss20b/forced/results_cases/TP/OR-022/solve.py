CSVQA_DATA = {'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesorders.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '52.4965'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'")
    I = [rec['data_row_number'] for rec in records]
    r_i = {}
    d_i = {}
    s_i = {}
    for rec in records:
        key = rec['data_row_number']
        values = rec.get('values', {})
        try:
            r_i[key] = float(values['Revenue'])
            d_i[key] = float(values['Demand'])
            s_i[key] = float(values['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing required field {e} in record {key}')
    for i in I:
        if i not in r_i:
            raise ValueError(f'Missing revenue for product {i}')
        if i not in d_i:
            raise ValueError(f'Missing demand for product {i}')
        if i not in s_i:
            raise ValueError(f'Missing inventory for product {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= d_i[i], name=f'demand_{i}')
        model.addConstr(x[i] <= s_i[i], name=f'inventory_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
