CSVQA_DATA = {'query': 'The company operates in the European market and offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'EuropeSalesRecords.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '765850', 'Initial Inventory': '5627060', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    product_names = []
    r_i = {}
    d_i = {}
    s_i = {}
    required_columns = {'Product Name', 'Revenue', 'Demand', 'Initial Inventory'}
    for rec in records:
        values = rec.get('values', {})
        if not required_columns.issubset(values.keys()):
            missing = required_columns - values.keys()
            raise ValueError(f"Missing columns in record {rec.get('data_row_number')}: {missing}")
        prod = values['Product Name']
        prod_key = str(prod)
        try:
            revenue = float(values['Revenue'])
            demand = float(values['Demand'])
            inventory = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Numeric conversion error in record {rec.get('data_row_number')}: {e}")
        product_names.append(prod_key)
        r_i[prod_key] = revenue
        d_i[prod_key] = demand
        s_i[prod_key] = inventory
    m = gurobipy.Model('Baby_Product_Revenue_Maximization')
    m.setParam('MIPGap', 1e-06)
    ub_dict = {i: min(d_i[i], s_i[i]) for i in product_names}
    x = m.addVars(product_names, lb=0, ub=ub_dict, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
