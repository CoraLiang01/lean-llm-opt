CSVQA_DATA = {'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘Organ’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Revenue', 'Demand', 'Initial Inventory', 'Sub Category'], 'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv', 'records': [{'data_row_number': 17, 'values': {'Demand': '678906', 'Initial Inventory': '5034020.0', 'Revenue': '60.8', 'Sub Category': 'Organic Fruits'}}, {'data_row_number': 18, 'values': {'Demand': '749927', 'Initial Inventory': '5589290.0', 'Revenue': '918.45', 'Sub Category': 'Organic Staples'}}, {'data_row_number': 19, 'values': {'Demand': '699808', 'Initial Inventory': '5202710.0', 'Revenue': '77.52', 'Sub Category': 'Organic Vegetables'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError('Required table file_0_view_0 not found in CSVQA_DATA')
    I = []
    rev = {}
    demand = {}
    inv = {}
    for record in table['records']:
        values = record['values']
        sub_cat = values.get('Sub Category', '')
        if 'Organ' not in sub_cat:
            continue
        data_row_number = record['data_row_number']
        I.append(data_row_number)
        try:
            rev_val = float(values['Revenue'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Revenue for data_row_number {data_row_number}')
        rev[data_row_number] = rev_val
        try:
            demand_val = int(values['Demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Demand for data_row_number {data_row_number}')
        demand[data_row_number] = demand_val
        try:
            inv_val = float(values['Initial Inventory'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Initial Inventory for data_row_number {data_row_number}')
        inv[data_row_number] = inv_val
    for i in I:
        if i not in rev or i not in demand or i not in inv:
            raise ValueError(f'Missing parameter for index {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in I), name='')
    m.addConstrs((x[i] <= inv[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
