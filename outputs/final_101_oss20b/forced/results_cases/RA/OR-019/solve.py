CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘27in’ product i that the company plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SalesDataAnalysis.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '149.99'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    if not tables:
        raise ValueError('No tables found in CSVQA_DATA')
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records in the required table')
    I = []
    rev = {}
    dem = {}
    init = {}
    required_columns = {'Product Name', 'Revenue', 'Demand', 'Initial Inventory'}
    for rec in records:
        values = rec.get('values', {})
        if not required_columns.issubset(values):
            missing = required_columns - values.keys()
            raise ValueError(f"Missing columns in record {rec.get('data_row_number')}: {missing}")
        prod_name = values['Product Name']
        try:
            revenue = float(values['Revenue'])
            demand = int(values['Demand'])
            inventory = int(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Non-numeric value in record {rec.get('data_row_number')}: {e}")
        I.append(prod_name)
        rev[prod_name] = revenue
        dem[prod_name] = demand
        init[prod_name] = inventory
    m = gurobipy.Model('27in_product_revenue_maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.addConstrs((x[i] <= dem[i] for i in I), name='')
    m.addConstrs((x[i] <= init[i] for i in I), name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
