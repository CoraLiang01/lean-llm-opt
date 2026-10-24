CSVQA_DATA = {'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RestaurantSalesreport.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1483', 'Initial Inventory': '10440.0', 'Product Name': 'Aalopuri', 'Revenue': '20'}}], 'table_id': 'file_0_view_0'}]}
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
    initial_inventory = {}
    product_names = []
    for record in records:
        values = record.get('values', {})
        prod = values.get('Product Name')
        if prod is None:
            raise ValueError("Record missing 'Product Name'")
        if not prod.startswith('Aalop'):
            continue
        rev = values.get('Revenue')
        dem = values.get('Demand')
        inv = values.get('Initial Inventory')
        if rev is None or dem is None or inv is None:
            raise ValueError(f'Missing parameter for product {prod}')
        try:
            rev_val = float(rev)
            dem_val = float(dem)
            inv_val = float(inv)
        except ValueError as e:
            raise ValueError(f'Non‑numeric parameter for product {prod}: {e}')
        revenue[prod] = rev_val
        demand[prod] = dem_val
        initial_inventory[prod] = inv_val
        product_names.append(prod)
    if not product_names:
        raise ValueError('No Aalop products found after filtering')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in product_names), name='')
    m.addConstrs((x[i] <= initial_inventory[i] for i in product_names), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
