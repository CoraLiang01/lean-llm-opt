CSVQA_DATA = {'query': 'A retail store is managing the sales of various product categories, with detailed revenue data available in the ‘Revenue’ column of the dataset. Each product category has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘ZZ’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘ZZ’ product i that the store plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['SKU', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RetailStoreSalesTransactions(ScannerData).csv', 'records': [{'data_row_number': 5237, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '24.38', 'SKU': 'ZZ2AO'}}, {'data_row_number': 5238, 'values': {'Demand': '4', 'Initial Inventory': '20.0', 'Revenue': '30.12', 'SKU': 'ZZDW7'}}, {'data_row_number': 5239, 'values': {'Demand': '82', 'Initial Inventory': '530.0', 'Revenue': '19.52', 'SKU': 'ZZM1A'}}, {'data_row_number': 5240, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '10.79', 'SKU': 'ZZNC5'}}, {'data_row_number': 5241, 'values': {'Demand': '2', 'Initial Inventory': '10.0', 'Revenue': '111.81', 'SKU': 'ZZX6K'}}], 'table_id': 'file_0_view_0'}]}
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
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        vals = rec.get('values', {})
        sku = vals.get('SKU')
        if sku is None:
            raise ValueError(f"Missing SKU in record {rec.get('data_row_number')}")
        rev_str = vals.get('Revenue')
        dem_str = vals.get('Demand')
        inv_str = vals.get('Initial Inventory')
        if rev_str is None or dem_str is None or inv_str is None:
            raise ValueError(f'Missing parameter in record for SKU {sku}')
        try:
            rev = float(rev_str)
            dem = int(dem_str)
            inv = float(inv_str)
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for SKU {sku}: {e}')
        revenue[sku] = rev
        demand[sku] = dem
        inventory[sku] = inv
    skus = set(revenue.keys())
    if not skus.issubset(demand.keys()) or not skus.issubset(inventory.keys()):
        missing = skus - demand.keys() | skus - inventory.keys()
        raise ValueError(f'Missing parameters for SKUs: {missing}')
    m = gurobipy.Model('ZZ_Product_Sales')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(skus, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[s] * x[s] for s in skus)), GRB.MAXIMIZE)
    m.addConstrs((x[s] <= demand[s] for s in skus), name='')
    m.addConstrs((x[s] <= inventory[s] for s in skus), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
