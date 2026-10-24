CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘27in’ product i that the company plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SalesDataAnalysis.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '149.99'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == table_id), None)
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    I = []
    rev_i = {}
    dem_i = {}
    inv_i = {}
    for record in table['records']:
        data_row_number = record['data_row_number']
        values = record['values']
        try:
            rev = float(values['Revenue'])
            dem = int(values['Demand'])
            inv = int(values['Initial Inventory'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid data in record {data_row_number}: {e}')
        I.append(data_row_number)
        rev_i[data_row_number] = rev
        dem_i[data_row_number] = dem
        inv_i[data_row_number] = inv
    for i in I:
        if i not in rev_i or i not in dem_i or i not in inv_i:
            raise ValueError(f'Missing parameters for index {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((rev_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= dem_i[i], name='')
        model.addConstr(x[i] <= inv_i[i], name='')
    model.optimize()
    return model
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective value: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.Status}')
