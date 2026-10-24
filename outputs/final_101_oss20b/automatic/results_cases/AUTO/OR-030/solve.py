CSVQA_DATA = {'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ column. The dealership aims to maximize total revenue using the initial inventory of car models classified under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'BigMartSales.csv', 'records': [{'data_row_number': 1163, 'values': {'Demand': '30', 'Initial Inventory': '200', 'Product Name': 'FDK57', 'Revenue': '119.144'}}, {'data_row_number': 1501, 'values': {'Demand': '40', 'Initial Inventory': '100', 'Product Name': 'FDK57', 'Revenue': '119.144'}}, {'data_row_number': 1576, 'values': {'Demand': '30', 'Initial Inventory': '200', 'Product Name': 'FDK57', 'Revenue': '121.244'}}, {'data_row_number': 1793, 'values': {'Demand': '50', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.144'}}, {'data_row_number': 2438, 'values': {'Demand': '10', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.544'}}, {'data_row_number': 4297, 'values': {'Demand': '30', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '121.244'}}, {'data_row_number': 4806, 'values': {'Demand': '50', 'Initial Inventory': '250', 'Product Name': 'FDK57', 'Revenue': '119.744'}}, {'data_row_number': 4942, 'values': {'Demand': '50', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.844'}}], 'table_id': 'file_0_view_0'}]}
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
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA.")
    records = target_table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'.")
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        src_row = rec.get('data_row_number')
        if src_row is None:
            raise ValueError("Record missing 'data_row_number' key.")
        vals = rec.get('values', {})
        try:
            rev = float(vals.get('Revenue', 0))
            dem = int(vals.get('Demand', 0))
            inv = int(vals.get('Initial Inventory', 0))
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid numeric value in record {src_row}: {e}')
        I.append(src_row)
        revenue[src_row] = rev
        demand[src_row] = dem
        inventory[src_row] = inv
    for i in I:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for index {i}.')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in I), name='')
    m.addConstrs((x[i] <= inventory[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
