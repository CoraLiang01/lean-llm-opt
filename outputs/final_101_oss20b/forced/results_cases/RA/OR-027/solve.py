CSVQA_DATA = {'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘Organ’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv', 'records': [{'data_row_number': 17, 'values': {'Demand': '678906', 'Initial Inventory': '5034020.0', 'Revenue': '60.8', 'Sub Category': 'Organic Fruits'}}, {'data_row_number': 18, 'values': {'Demand': '749927', 'Initial Inventory': '5589290.0', 'Revenue': '918.45', 'Sub Category': 'Organic Staples'}}, {'data_row_number': 19, 'values': {'Demand': '699808', 'Initial Inventory': '5202710.0', 'Revenue': '77.52', 'Sub Category': 'Organic Vegetables'}}], 'table_id': 'file_0_view_0'}]}
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
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    I = []
    revenue_dict = {}
    demand_dict = {}
    inventory_dict = {}
    for rec in records:
        data_row_number = rec.get('data_row_number')
        if data_row_number is None:
            raise ValueError("Record missing 'data_row_number' key")
        values = rec.get('values', {})
        for col in ('Revenue', 'Demand', 'Initial Inventory'):
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record {data_row_number}")
        try:
            rev = float(values['Revenue'])
            dem = int(values['Demand'])
            inv = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {data_row_number}: {e}')
        I.append(data_row_number)
        revenue_dict[data_row_number] = rev
        demand_dict[data_row_number] = dem
        inventory_dict[data_row_number] = inv
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand_dict[i] for i in I), name='')
    m.addConstrs((x[i] <= inventory_dict[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
