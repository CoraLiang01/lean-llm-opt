CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.', 'relationships': [], 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'OnlineRetailSalesDataset.csv', 'records': [{'data_row_number': 899, 'values': {'Demand': '8171', 'Initial Inventory': '56450', 'Revenue': '434.74', 'id_number': 'id999'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0'))
    records = table['records']
    I = []
    r = {}
    d = {}
    inv = {}
    for rec in records:
        values = rec['values']
        prod_id = values['id_number']
        I.append(prod_id)
        try:
            r[prod_id] = float(values['Revenue'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Revenue for {prod_id}')
        try:
            d[prod_id] = int(values['Demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Demand for {prod_id}')
        try:
            inv[prod_id] = int(values['Initial Inventory'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Initial Inventory for {prod_id}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= inv[i] for i in I), name='')
    model.addConstrs((x[i] <= d[i] for i in I), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
