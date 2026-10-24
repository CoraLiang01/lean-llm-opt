CSVQA_DATA = {'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘Organ’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv', 'records': [{'data_row_number': 17, 'values': {'Demand': '678906', 'Initial Inventory': '5034020.0', 'Revenue': '60.8', 'Sub Category': 'Organic Fruits'}}, {'data_row_number': 18, 'values': {'Demand': '749927', 'Initial Inventory': '5589290.0', 'Revenue': '918.45', 'Sub Category': 'Organic Staples'}}, {'data_row_number': 19, 'values': {'Demand': '699808', 'Initial Inventory': '5202710.0', 'Revenue': '77.52', 'Sub Category': 'Organic Vegetables'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == table_id), None)
    if table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'")
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        key = rec.get('data_row_number')
        if key is None:
            raise ValueError("Record missing 'data_row_number' key")
        values = rec.get('values', {})
        for col in ('Revenue', 'Demand', 'Initial Inventory'):
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record with data_row_number {key}")
        try:
            rev = float(values['Revenue'])
            dem = int(float(values['Demand']))
            inv = float(values['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Error converting values for data_row_number {key}: {e}')
        revenue[key] = rev
        demand[key] = dem
        inventory[key] = inv
    I = list(revenue.keys())
    m = gurobipy.Model('Organ_Product_Revenue_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
