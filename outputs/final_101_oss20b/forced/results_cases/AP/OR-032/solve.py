CSVQA_DATA = {'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'DifferentStoreSales.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1980', 'Initial Inventory': '9920.0', 'Product_Name': 'Books_15.15', 'Revenue': '15.15'}}, {'data_row_number': 1, 'values': {'Demand': '3024', 'Initial Inventory': '20160.0', 'Product_Name': 'Books_30.3', 'Revenue': '30.3'}}, {'data_row_number': 2, 'values': {'Demand': '4536', 'Initial Inventory': '30000.0', 'Product_Name': 'Books_45.45', 'Revenue': '45.45'}}, {'data_row_number': 3, 'values': {'Demand': '5601', 'Initial Inventory': '38360.0', 'Product_Name': 'Books_60.6', 'Revenue': '60.6'}}, {'data_row_number': 4, 'values': {'Demand': '7567', 'Initial Inventory': '51450.0', 'Product_Name': 'Books_75.75', 'Revenue': '75.75'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for t in CSVQA_DATA['tables']:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError('Table file_0_view_0 not found')
    records = table.get('records', [])
    I = []
    r_i = {}
    d_i = {}
    v_i = {}
    for rec in records:
        values = rec.get('values', {})
        prod = values.get('Product_Name')
        if prod is None:
            raise ValueError('Missing Product_Name in record')
        if 'Revenue' not in values or 'Demand' not in values or 'Initial Inventory' not in values:
            raise ValueError(f'Missing parameter in record for product {prod}')
        I.append(prod)
        try:
            r_i[prod] = float(values['Revenue'])
            d_i[prod] = float(values['Demand'])
            v_i[prod] = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record for product {prod}') from e
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d_i[i], name=f'c_demand_{i}')
        m.addConstr(x[i] <= v_i[i], name=f'c_inventory_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
