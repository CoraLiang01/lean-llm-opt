CSVQA_DATA = {'query': 'The company operates in the European market and offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'EuropeSalesRecords.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '765850', 'Initial Inventory': '5627060', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    target_table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            target_table = t
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA.")
    records = target_table.get('records', [])
    if not records:
        raise ValueError("No records returned for table 'file_0_view_0'.")
    R_i = {}
    D_i = {}
    I_i = {}
    for rec in records:
        vals = rec.get('values', {})
        try:
            product = vals['Product Name']
            revenue = float(vals['Revenue'])
            demand = int(vals['Demand'])
            inventory = int(vals['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f"Missing required column in record {rec.get('data_row_number')}: {e}")
        if product in R_i:
            raise ValueError(f'Duplicate product key detected: {product}')
        R_i[product] = revenue
        D_i[product] = demand
        I_i[product] = inventory
    keys = set(R_i.keys())
    if keys != set(D_i.keys()) or keys != set(I_i.keys()):
        raise ValueError('Inconsistent product keys across parameters.')
    I = list(keys)
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.addConstrs((x[i] <= D_i[i] for i in I), name='')
    m.addConstrs((x[i] <= I_i[i] for i in I), name='')
    m.setObjective(gurobipy.quicksum((R_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
