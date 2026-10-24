CSVQA_DATA = {'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread to order each day to maximize our total expected profit while staying within our storage limits. The decision variables x_i represents the number of units of bread type i to be ordered each day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '180'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'data_row_number': 1, 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'data_row_number': 2, 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'data_row_number': 3, 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'data_row_number': 4, 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'data_row_number': 5, 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'data_row_number': 6, 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'data_row_number': 7, 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'data_row_number': 8, 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'data_row_number': 9, 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    if capacity_table is None:
        raise ValueError('Capacity table not found')
    if not capacity_table['records']:
        raise ValueError('Capacity table has no records')
    capacity_record = capacity_table['records'][0]
    try:
        C = float(capacity_record['values']['Capacity'])
    except (KeyError, ValueError, TypeError):
        raise ValueError('Capacity value missing or invalid')
    products_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'), None)
    if products_table is None:
        raise ValueError('Products table not found')
    if not products_table['records']:
        raise ValueError('Products table has no records')
    I = []
    v_i = {}
    w_i = {}
    for rec in products_table['records']:
        try:
            prod = rec['values']['ProductName']
            val = float(rec['values']['Value'])
            wt = float(rec['values']['Weight'])
        except KeyError as e:
            raise ValueError(f'Missing field in product record: {e}')
        if prod in I:
            raise ValueError(f'Duplicate product name: {prod}')
        I.append(prod)
        v_i[prod] = val
        w_i[prod] = wt
    for prod in I:
        if prod not in v_i or prod not in w_i:
            raise ValueError(f'Missing parameters for product {prod}')
    m = gurobipy.Model('Bakery_Profit_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((v_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w_i[i] * x[i] for i in I)) <= C, name='storage')
    m.optimize()
    return m
m = solve_problem()
