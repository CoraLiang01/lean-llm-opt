CSVQA_DATA = {'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RestaurantSalesreport.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1483', 'Initial Inventory': '10440.0', 'Product Name': 'Aalopuri', 'Revenue': '20'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    records = target_table.get('records', [])
    if not records:
        raise ValueError(f'No records found in table {table_id}')
    revenue_dict = {}
    demand_dict = {}
    inventory_dict = {}
    for rec in records:
        values = rec.get('values', {})
        prod_name = values.get('Product Name')
        if prod_name is None:
            raise ValueError('Missing Product Name in record')
        try:
            rev = float(values.get('Revenue', 0))
            dem = float(values.get('Demand', 0))
            inv = float(values.get('Initial Inventory', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f'Invalid numeric value in record for product {prod_name}') from e
        if prod_name in revenue_dict:
            raise ValueError(f'Duplicate product name detected: {prod_name}')
        revenue_dict[prod_name] = rev
        demand_dict[prod_name] = dem
        inventory_dict[prod_name] = inv
    for prod in revenue_dict:
        if prod not in demand_dict or prod not in inventory_dict:
            raise ValueError(f'Missing parameter for product {prod}')
    m = gurobipy.Model('AalopRevenueMax')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(revenue_dict.keys(), lb=0, vtype=GRB.INTEGER, name='')
    m.addConstrs((x[i] <= demand_dict[i] for i in revenue_dict.keys()), name='')
    m.addConstrs((x[i] <= inventory_dict[i] for i in revenue_dict.keys()), name='')
    m.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in revenue_dict.keys())), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
