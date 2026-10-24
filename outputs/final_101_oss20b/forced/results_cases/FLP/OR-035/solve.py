CSVQA_DATA = {'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread to order each day to maximize our total expected profit while staying within our storage limits. The decision variables x_i represents the number of units of bread type i to be ordered each day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '180'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'data_row_number': 1, 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'data_row_number': 2, 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'data_row_number': 3, 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'data_row_number': 4, 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'data_row_number': 5, 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'data_row_number': 6, 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'data_row_number': 7, 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'data_row_number': 8, 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'data_row_number': 9, 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    cap_table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if cap_table is None or not cap_table.get('records'):
        raise ValueError("Capacity table 'file_0_view_0' missing or empty.")
    cap_record = cap_table['records'][0]['values']
    if 'Capacity' not in cap_record:
        raise ValueError('Capacity field missing in capacity record.')
    try:
        C = int(cap_record['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid capacity value: {cap_record['Capacity']}") from e
    prod_table = next((t for t in tables if t.get('table_id') == 'file_1_view_0'), None)
    if prod_table is None or not prod_table.get('records'):
        raise ValueError("Products table 'file_1_view_0' missing or empty.")
    product_names = []
    v_i = {}
    w_i = {}
    for rec in prod_table['records']:
        vals = rec['values']
        if 'ProductName' not in vals or 'Value' not in vals or 'Weight' not in vals:
            raise ValueError(f'Missing fields in product record: {rec}')
        pname = vals['ProductName']
        try:
            val = int(vals['Value'])
            wt = int(vals['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid numeric value in product record {pname}') from e
        product_names.append(pname)
        v_i[pname] = val
        w_i[pname] = wt
    if not product_names:
        raise ValueError('No products found in products table.')
    m = gurobipy.Model('Bakery_Profit_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((v_i[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w_i[p] * x[p] for p in product_names)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
