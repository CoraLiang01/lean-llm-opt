CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘27in’ product i that the company plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SalesDataAnalysis.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '149.99'}}], 'table_id': 'file_0_view_0'}]}
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
    index_set = []
    for rec in records:
        values = rec.get('values', {})
        product_name = values.get('Product Name')
        if product_name is None:
            raise ValueError('Missing Product Name in record')
        try:
            revenue = float(values.get('Revenue', 0))
            demand = int(values.get('Demand', 0))
            inventory = int(values.get('Initial Inventory', 0))
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid numeric value for product {product_name}: {e}')
        if product_name in revenue_dict:
            raise ValueError(f'Duplicate product name detected: {product_name}')
        revenue_dict[product_name] = revenue
        demand_dict[product_name] = demand
        inventory_dict[product_name] = inventory
        index_set.append(product_name)
    for i in index_set:
        if i not in revenue_dict or i not in demand_dict or i not in inventory_dict:
            raise ValueError(f'Missing parameter for product {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(index_set, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in index_set)), GRB.MAXIMIZE)
    for i in index_set:
        m.addConstr(x[i] <= demand_dict[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory_dict[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()
