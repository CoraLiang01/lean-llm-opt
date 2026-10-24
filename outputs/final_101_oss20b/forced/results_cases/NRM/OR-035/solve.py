CSVQA_DATA = {'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread to order each day to maximize our total expected profit while staying within our storage limits. The decision variables x_i represents the number of units of bread type i to be ordered each day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '180'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'data_row_number': 1, 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'data_row_number': 2, 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'data_row_number': 3, 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'data_row_number': 4, 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'data_row_number': 5, 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'data_row_number': 6, 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'data_row_number': 7, 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'data_row_number': 8, 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'data_row_number': 9, 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    tables = data['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    product_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    capacity_record = capacity_table['records'][0]
    capacity = float(capacity_record['values']['Capacity'])
    product_names = []
    profit = {}
    weight = {}
    for rec in product_table['records']:
        name = rec['values']['ProductName']
        product_names.append(name)
        profit[name] = float(rec['values']['Value'])
        weight[name] = float(rec['values']['Weight'])
    for name in product_names:
        if name not in profit or name not in weight:
            raise ValueError(f'Missing data for product {name}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((profit[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight[i] * x[i] for i in product_names)) <= capacity, name='capacity')
    m.optimize()
    return m
m = solve_problem()
