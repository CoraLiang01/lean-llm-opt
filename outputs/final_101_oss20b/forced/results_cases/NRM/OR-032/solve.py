CSVQA_DATA = {'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'DifferentStoreSales.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1980', 'Initial Inventory': '9920.0', 'Product_Name': 'Books_15.15', 'Revenue': '15.15'}}, {'data_row_number': 1, 'values': {'Demand': '3024', 'Initial Inventory': '20160.0', 'Product_Name': 'Books_30.3', 'Revenue': '30.3'}}, {'data_row_number': 2, 'values': {'Demand': '4536', 'Initial Inventory': '30000.0', 'Product_Name': 'Books_45.45', 'Revenue': '45.45'}}, {'data_row_number': 3, 'values': {'Demand': '5601', 'Initial Inventory': '38360.0', 'Product_Name': 'Books_60.6', 'Revenue': '60.6'}}, {'data_row_number': 4, 'values': {'Demand': '7567', 'Initial Inventory': '51450.0', 'Product_Name': 'Books_75.75', 'Revenue': '75.75'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec['values']
        prod = values['Product_Name']
        try:
            rev = float(values['Revenue'])
        except (ValueError, TypeError):
            raise ValueError(f'Invalid revenue for product {prod}')
        try:
            dem = int(values['Demand'])
        except (ValueError, TypeError):
            raise ValueError(f'Invalid demand for product {prod}')
        try:
            inv = float(values['Initial Inventory'])
        except (ValueError, TypeError):
            raise ValueError(f'Invalid inventory for product {prod}')
        revenue[prod] = rev
        demand[prod] = dem
        inventory[prod] = inv
    prod_set = set(revenue.keys())
    if prod_set != set(demand.keys()) or prod_set != set(inventory.keys()):
        missing = prod_set.symmetric_difference(demand.keys()).union(prod_set.symmetric_difference(inventory.keys()))
        raise ValueError(f'Missing parameters for products: {missing}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(prod_set, lb=1, vtype=GRB.INTEGER, name='')
    model.addConstrs((x[i] <= demand[i] for i in prod_set), name='')
    model.addConstrs((x[i] <= inventory[i] for i in prod_set), name='')
    model.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in prod_set)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    return model
m = solve_problem()
