CSVQA_DATA = {'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘ZZ’ product i that the store plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RetailStoreSalesTransactions(ScannerData).csv', 'records': [{'data_row_number': 5237, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}}, {'data_row_number': 5238, 'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}}, {'data_row_number': 5239, 'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}}, {'data_row_number': 5240, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}}, {'data_row_number': 5241, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError("No records in table 'file_0_view_0'")
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec.get('values', {})
        sku = values.get('SKU')
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if sku is None or rev_str is None or dem_str is None or (inv_str is None):
            raise ValueError(f"Missing required field in record {rec.get('data_row_number')}")
        try:
            rev = float(rev_str)
        except Exception as e:
            raise ValueError(f"Invalid revenue value '{rev_str}' for SKU {sku}") from e
        try:
            dem = int(dem_str)
        except Exception as e:
            raise ValueError(f"Invalid demand value '{dem_str}' for SKU {sku}") from e
        try:
            inv = float(inv_str)
        except Exception as e:
            raise ValueError(f"Invalid inventory value '{inv_str}' for SKU {sku}") from e
        I.append(sku)
        revenue[sku] = rev
        demand[sku] = dem
        inventory[sku] = inv
    ub_dict = {sku: min(demand[sku], inventory[sku]) for sku in I}
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, ub=ub_dict, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue[sku] * x[sku] for sku in I)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
