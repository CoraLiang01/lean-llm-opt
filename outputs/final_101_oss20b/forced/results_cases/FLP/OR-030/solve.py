CSVQA_DATA = {'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ column. The dealership aims to maximize total revenue using the initial inventory of car models classified under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'BigMartSales.csv', 'records': [{'data_row_number': 1163, 'values': {'Demand': '30', 'Initial Inventory': '200', 'Product Name': 'FDK57', 'Revenue': '119.144'}}, {'data_row_number': 1501, 'values': {'Demand': '40', 'Initial Inventory': '100', 'Product Name': 'FDK57', 'Revenue': '119.144'}}, {'data_row_number': 1576, 'values': {'Demand': '30', 'Initial Inventory': '200', 'Product Name': 'FDK57', 'Revenue': '121.244'}}, {'data_row_number': 1793, 'values': {'Demand': '50', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.144'}}, {'data_row_number': 2438, 'values': {'Demand': '10', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.544'}}, {'data_row_number': 4297, 'values': {'Demand': '30', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '121.244'}}, {'data_row_number': 4806, 'values': {'Demand': '50', 'Initial Inventory': '250', 'Product Name': 'FDK57', 'Revenue': '119.744'}}, {'data_row_number': 4942, 'values': {'Demand': '50', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.844'}}], 'table_id': 'file_0_view_0'}]}
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
        raise ValueError('Table file_0_view_0 not found in CSVQA_DATA')
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records in target table')
    r = {}
    d = {}
    inv = {}
    for rec in records:
        i = rec.get('data_row_number')
        if i is None:
            raise ValueError('Record missing data_row_number')
        vals = rec.get('values', {})
        for col in ('Revenue', 'Demand', 'Initial Inventory'):
            if col not in vals:
                raise ValueError(f'Missing column {col} in record {i}')
        try:
            r[i] = float(vals['Revenue'])
            d[i] = float(vals['Demand'])
            inv[i] = float(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Non-numeric value in record {i}: {e}')
    ids = list(r.keys())
    if set(ids) != set(d.keys()) or set(ids) != set(inv.keys()):
        raise ValueError('Parameter key mismatch among r, d, inv')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((r[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    for i in ids:
        model.addConstr(x[i] <= inv[i], name=f'inv_{i}')
        model.addConstr(x[i] <= d[i], name=f'demand_{i}')
    model.optimize()
    return model
m = solve_problem()
