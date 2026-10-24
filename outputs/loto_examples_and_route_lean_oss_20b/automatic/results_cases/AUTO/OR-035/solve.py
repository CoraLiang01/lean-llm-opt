CSVQA_DATA = {'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread to order each day to maximize our total expected profit while staying within our storage limits. The decision variables x_i represents the number of units of bread type i to be ordered each day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '180'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'data_row_number': 1, 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'data_row_number': 2, 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'data_row_number': 3, 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'data_row_number': 4, 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'data_row_number': 5, 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'data_row_number': 6, 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'data_row_number': 7, 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'data_row_number': 8, 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'data_row_number': 9, 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'), None)
    if capacity_table is None or not capacity_table['records']:
        raise ValueError('Capacity table missing or empty')
    capacity_value = capacity_table['records'][0]['values']['Capacity']
    try:
        C = int(capacity_value)
    except Exception as e:
        raise ValueError(f'Invalid capacity value: {capacity_value}') from e
    products_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'), None)
    if products_table is None or not products_table['records']:
        raise ValueError('Products table missing or empty')
    records = products_table['records']
    I = list(range(1, len(records) + 1))
    v = {}
    w = {}
    for idx, rec in enumerate(records, start=1):
        try:
            v_val = int(rec['values']['Value'])
            w_val = int(rec['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid product data at record {idx}') from e
        v[idx] = v_val
        w[idx] = w_val
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w[i] * x[i] for i in I)) <= C, name='Capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Status: {m.Status}')
    return m
m = solve_problem()
