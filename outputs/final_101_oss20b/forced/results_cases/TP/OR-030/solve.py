CSVQA_DATA = {'query': 'A car dealership manages the sales of various car models with revenue data provided in the ‘Revenue’ column. The dealership aims to maximize total revenue using the initial inventory of car models classified under ‘FDK57’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the quantity of each ‘FDK57’ car model i that the dealership plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'BigMartSales.csv', 'records': [{'data_row_number': 1163, 'values': {'Demand': '30', 'Initial Inventory': '200', 'Product Name': 'FDK57', 'Revenue': '119.144'}}, {'data_row_number': 1501, 'values': {'Demand': '40', 'Initial Inventory': '100', 'Product Name': 'FDK57', 'Revenue': '119.144'}}, {'data_row_number': 1576, 'values': {'Demand': '30', 'Initial Inventory': '200', 'Product Name': 'FDK57', 'Revenue': '121.244'}}, {'data_row_number': 1793, 'values': {'Demand': '50', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.144'}}, {'data_row_number': 2438, 'values': {'Demand': '10', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.544'}}, {'data_row_number': 4297, 'values': {'Demand': '30', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '121.244'}}, {'data_row_number': 4806, 'values': {'Demand': '50', 'Initial Inventory': '250', 'Product Name': 'FDK57', 'Revenue': '119.744'}}, {'data_row_number': 4942, 'values': {'Demand': '50', 'Initial Inventory': '150', 'Product Name': 'FDK57', 'Revenue': '120.844'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    r = {}
    d = {}
    s = {}
    for rec in table['records']:
        i = rec['data_row_number']
        vals = rec['values']
        try:
            r[i] = float(vals['Revenue'])
            d[i] = float(vals['Demand'])
            s[i] = float(vals['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing required column in record {i}: {e}')
    indices = set(r.keys())
    if not indices.issubset(d.keys()) or not indices.issubset(s.keys()):
        missing = indices - d.keys() | indices - s.keys()
        raise ValueError(f'Missing data for indices: {missing}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(indices, lb=0, vtype=GRB.CONTINUOUS, name='')
    for i in indices:
        m.addConstr(x[i] <= d[i])
        m.addConstr(x[i] <= s[i])
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in indices)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
