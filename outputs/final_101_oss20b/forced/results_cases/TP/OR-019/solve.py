CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘27in’ product i that the company plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SalesDataAnalysis.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '149.99'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec.get('values', {})
        for col in ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']:
            if col not in values:
                raise ValueError(f"Missing column '{col}' in record {rec.get('data_row_number')}")
        prod_name = values['Product Name']
        I.append(prod_name)
        try:
            revenue[prod_name] = float(values['Revenue'])
            demand[prod_name] = float(values['Demand'])
            inventory[prod_name] = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Non-numeric value in record {rec.get('data_row_number')}: {e}")
    m = gurobipy.Model('MaximizeRevenue27in')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    for i in I:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
