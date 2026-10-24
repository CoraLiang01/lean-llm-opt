CSVQA_DATA = {'query': 'An e-commerce platform sells women’s clothing items with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘FAUX’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘FAUX’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Initial Inventory', 'Demand'], 'file_name': 'ZARASales.csv', 'records': [{'data_row_number': 56, 'values': {'Demand': '3025', 'Initial Inventory': '20970', 'Product Name': 'FAUX FUR JEWEL SWEATER', 'Revenue': '35.9'}}, {'data_row_number': 57, 'values': {'Demand': '9585', 'Initial Inventory': '71970', 'Product Name': 'FAUX LEATHER BOMBER JACKET', 'Revenue': '69.9'}}, {'data_row_number': 58, 'values': {'Demand': '4486', 'Initial Inventory': '32730', 'Product Name': 'FAUX LEATHER BOXY FIT JACKET', 'Revenue': '99.9'}}, {'data_row_number': 59, 'values': {'Demand': '10322', 'Initial Inventory': '71130', 'Product Name': 'FAUX LEATHER JACKET', 'Revenue': '99.9'}}, {'data_row_number': 60, 'values': {'Demand': '4868', 'Initial Inventory': '34910', 'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION', 'Revenue': '159.0'}}, {'data_row_number': 61, 'values': {'Demand': '8482', 'Initial Inventory': '64010', 'Product Name': 'FAUX LEATHER PUFFER JACKET', 'Revenue': '69.99'}}, {'data_row_number': 62, 'values': {'Demand': '2607', 'Initial Inventory': '20760', 'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS', 'Revenue': '99.9'}}, {'data_row_number': 63, 'values': {'Demand': '1784', 'Initial Inventory': '12490', 'Product Name': 'FAUX SHEARLING PLAID JACKET', 'Revenue': '89.9'}}, {'data_row_number': 64, 'values': {'Demand': '6626', 'Initial Inventory': '50300', 'Product Name': 'FAUX SUEDE BOMBER JACKET', 'Revenue': '69.9'}}, {'data_row_number': 65, 'values': {'Demand': '3256', 'Initial Inventory': '24570', 'Product Name': 'FAUX SUEDE JACKET', 'Revenue': '89.9'}}, {'data_row_number': 66, 'values': {'Demand': '2955', 'Initial Inventory': '24430', 'Product Name': 'FAUX SUEDE OVERSHIRT', 'Revenue': '69.9'}}, {'data_row_number': 67, 'values': {'Demand': '910', 'Initial Inventory': '7070', 'Product Name': 'FAUX SUEDE PATCH JACKET', 'Revenue': '89.9'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError("No records found in table 'file_0_view_0'")
    revenue_dict = {}
    init_inv_dict = {}
    demand_dict = {}
    for rec in records:
        values = rec.get('values', {})
        for key in ('Product Name', 'Revenue', 'Initial Inventory', 'Demand'):
            if key not in values:
                raise ValueError(f"Missing required field '{key}' in record {rec.get('data_row_number')}")
        name = values['Product Name']
        try:
            rev = float(values['Revenue'])
            init_inv = int(values['Initial Inventory'])
            dem = int(values['Demand'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('data_row_number')}: {e}")
        revenue_dict[name] = rev
        init_inv_dict[name] = init_inv
        demand_dict[name] = dem
    I = list(revenue_dict.keys())
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= init_inv_dict[i])
        model.addConstr(x[i] <= demand_dict[i])
    model.optimize()
    return model
m = solve_problem()
