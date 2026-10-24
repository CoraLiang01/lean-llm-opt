CSVQA_DATA = {'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘Organ’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv', 'records': [{'data_row_number': 17, 'values': {'Demand': '678906', 'Initial Inventory': '5034020.0', 'Revenue': '60.8', 'Sub Category': 'Organic Fruits'}}, {'data_row_number': 18, 'values': {'Demand': '749927', 'Initial Inventory': '5589290.0', 'Revenue': '918.45', 'Sub Category': 'Organic Staples'}}, {'data_row_number': 19, 'values': {'Demand': '699808', 'Initial Inventory': '5202710.0', 'Revenue': '77.52', 'Sub Category': 'Organic Vegetables'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB
table = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0':
        table = tbl
        break
if table is None:
    raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
records = table.get('records', [])
if not records:
    raise ValueError("No records found in table 'file_0_view_0'")
R = {}
D = {}
I = {}
for rec in records:
    vals = rec.get('values', {})
    subcat = vals.get('Sub Category')
    if subcat is None:
        raise ValueError("Missing 'Sub Category' in record")
    try:
        revenue = float(vals['Revenue'])
    except (KeyError, ValueError):
        raise ValueError(f"Missing or invalid 'Revenue' for {subcat}")
    try:
        demand = int(vals['Demand'])
    except (KeyError, ValueError):
        raise ValueError(f"Missing or invalid 'Demand' for {subcat}")
    try:
        init_inv = float(vals['Initial Inventory'])
    except (KeyError, ValueError):
        raise ValueError(f"Missing or invalid 'Initial Inventory' for {subcat}")
    R[subcat] = revenue
    D[subcat] = demand
    I[subcat] = init_inv
I_set = list(R.keys())
m = gurobipy.Model('Organ_Product_Revenue_Maximization')
m.setParam('MIPGap', 1e-06)
x = m.addVars(I_set, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gurobipy.quicksum((R[i] * x[i] for i in I_set)), GRB.MAXIMIZE)
for i in I_set:
    m.addConstr(x[i] <= D[i], name=f'demand_{i}')
    m.addConstr(x[i] <= I[i], name=f'inventory_{i}')
m.optimize()
m = m
