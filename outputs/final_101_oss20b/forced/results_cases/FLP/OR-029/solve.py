CSVQA_DATA = {'query': 'An e-commerce platform sells women’s clothing items with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘FAUX’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘FAUX’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'ZARASales.csv', 'records': [{'data_row_number': 56, 'values': {'Demand': '3025', 'Initial Inventory': '20970', 'Product Name': 'FAUX FUR JEWEL SWEATER', 'Revenue': '35.9'}}, {'data_row_number': 57, 'values': {'Demand': '9585', 'Initial Inventory': '71970', 'Product Name': 'FAUX LEATHER BOMBER JACKET', 'Revenue': '69.9'}}, {'data_row_number': 58, 'values': {'Demand': '4486', 'Initial Inventory': '32730', 'Product Name': 'FAUX LEATHER BOXY FIT JACKET', 'Revenue': '99.9'}}, {'data_row_number': 59, 'values': {'Demand': '10322', 'Initial Inventory': '71130', 'Product Name': 'FAUX LEATHER JACKET', 'Revenue': '99.9'}}, {'data_row_number': 60, 'values': {'Demand': '4868', 'Initial Inventory': '34910', 'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION', 'Revenue': '159.0'}}, {'data_row_number': 61, 'values': {'Demand': '8482', 'Initial Inventory': '64010', 'Product Name': 'FAUX LEATHER PUFFER JACKET', 'Revenue': '69.99'}}, {'data_row_number': 62, 'values': {'Demand': '2607', 'Initial Inventory': '20760', 'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS', 'Revenue': '99.9'}}, {'data_row_number': 63, 'values': {'Demand': '1784', 'Initial Inventory': '12490', 'Product Name': 'FAUX SHEARLING PLAID JACKET', 'Revenue': '89.9'}}, {'data_row_number': 64, 'values': {'Demand': '6626', 'Initial Inventory': '50300', 'Product Name': 'FAUX SUEDE BOMBER JACKET', 'Revenue': '69.9'}}, {'data_row_number': 65, 'values': {'Demand': '3256', 'Initial Inventory': '24570', 'Product Name': 'FAUX SUEDE JACKET', 'Revenue': '89.9'}}, {'data_row_number': 66, 'values': {'Demand': '2955', 'Initial Inventory': '24430', 'Product Name': 'FAUX SUEDE OVERSHIRT', 'Revenue': '69.9'}}, {'data_row_number': 67, 'values': {'Demand': '910', 'Initial Inventory': '7070', 'Product Name': 'FAUX SUEDE PATCH JACKET', 'Revenue': '89.9'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    tables = data['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    product_names = []
    R = {}
    D = {}
    I = {}
    for rec in records:
        values = rec['values']
        pname = values['Product Name']
        product_names.append(pname)
        R[pname] = float(values['Revenue'])
        D[pname] = int(values['Demand'])
        I[pname] = int(values['Initial Inventory'])
    for pname in product_names:
        if pname not in R or pname not in D or pname not in I:
            raise ValueError(f'Missing parameter for {pname}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_names, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((R[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
    for p in product_names:
        model.addConstr(x[p] <= D[p], name=f'Demand_{p}')
        model.addConstr(x[p] <= I[p], name=f'Inventory_{p}')
    model.optimize()
    return model
m = solve_problem()
