CSVQA_DATA = {'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘Organ’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv', 'records': [{'data_row_number': 17, 'values': {'Demand': '678906', 'Initial Inventory': '5034020.0', 'Revenue': '60.8', 'Sub Category': 'Organic Fruits'}}, {'data_row_number': 18, 'values': {'Demand': '749927', 'Initial Inventory': '5589290.0', 'Revenue': '918.45', 'Sub Category': 'Organic Staples'}}, {'data_row_number': 19, 'values': {'Demand': '699808', 'Initial Inventory': '5202710.0', 'Revenue': '77.52', 'Sub Category': 'Organic Vegetables'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    I = []
    revenue = {}
    demand = {}
    inv = {}
    for rec in records:
        values = rec['values']
        subcat = values['Sub Category']
        I.append(subcat)
        try:
            revenue[subcat] = float(values['Revenue'])
            demand[subcat] = float(values['Demand'])
            inv[subcat] = float(values['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing parameter {e.args[0]} for {subcat}')
    for i in I:
        if i not in revenue:
            raise ValueError(f'Missing Revenue for {i}')
        if i not in demand:
            raise ValueError(f'Missing Demand for {i}')
        if i not in inv:
            raise ValueError(f'Missing Initial Inventory for {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= demand[i] for i in I), name='')
    model.addConstrs((x[i] <= inv[i] for i in I), name='')
    model.optimize()
    return model
m = solve_problem()
