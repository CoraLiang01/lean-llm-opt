CSVQA_DATA = {'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘ZZ’ product i that the store plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RetailStoreSalesTransactions(ScannerData).csv', 'records': [{'data_row_number': 5237, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}}, {'data_row_number': 5238, 'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}}, {'data_row_number': 5239, 'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}}, {'data_row_number': 5240, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}}, {'data_row_number': 5241, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == table_id), None)
    if table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA.")
    records = table.get('records', [])
    if not records:
        raise ValueError(f"No records returned for table '{table_id}'.")
    required_cols = {'SKU', 'Revenue', 'Demand', 'Initial Inventory'}
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec.get('values', {})
        if not required_cols.issubset(values.keys()):
            missing = required_cols - values.keys()
            raise ValueError(f"Record {rec.get('data_row_number')} missing columns: {missing}")
        sku = str(values['SKU']).strip()
        if sku in revenue:
            raise ValueError(f"Duplicate SKU '{sku}' found in records.")
        try:
            rev_val = float(values['Revenue'])
        except Exception as e:
            raise ValueError(f"Invalid Revenue value for SKU '{sku}': {values['Revenue']}") from e
        try:
            dem_val = int(values['Demand'])
        except Exception as e:
            raise ValueError(f"Invalid Demand value for SKU '{sku}': {values['Demand']}") from e
        try:
            inv_val = float(values['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Invalid Initial Inventory value for SKU '{sku}': {values['Initial Inventory']}") from e
        revenue[sku] = rev_val
        demand[sku] = dem_val
        inventory[sku] = inv_val
    I = list(revenue.keys())
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= demand[i] for i in I), name='')
    model.addConstrs((x[i] <= inventory[i] for i in I), name='')
    model.optimize()
    return model
m = solve_problem()
