CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.', 'relationships': [], 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'OnlineRetailSalesDataset.csv', 'records': [{'data_row_number': 899, 'values': {'Demand': '8171', 'Initial Inventory': '56450', 'Revenue': '434.74', 'id_number': 'id999'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    I = []
    r_i = {}
    d_i = {}
    s_i = {}
    for rec in records:
        values = rec.get('values', {})
        for col in ('id_number', 'Revenue', 'Demand', 'Initial Inventory'):
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record {rec.get('data_row_number')}")
        id_number = values['id_number']
        if not id_number.startswith('id999'):
            continue
        try:
            revenue = float(values['Revenue'])
            demand = int(values['Demand'])
            inventory = int(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('data_row_number')}: {e}")
        I.append(id_number)
        r_i[id_number] = revenue
        d_i[id_number] = demand
        s_i[id_number] = inventory
    if not I:
        raise ValueError("No product identifiers starting with 'id999' found")
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    model.addConstrs((x[i] <= d_i[i] for i in I), name='')
    model.addConstrs((x[i] <= s_i[i] for i in I), name='')
    model.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    return model
m = solve_problem()
