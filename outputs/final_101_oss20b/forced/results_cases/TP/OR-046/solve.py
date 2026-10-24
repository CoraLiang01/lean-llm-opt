CSVQA_DATA = {'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered each day.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '875'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Weight', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Spinach', 'Value': '64', 'Weight': '230'}}, {'data_row_number': 1, 'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '75', 'Weight': '637'}}, {'data_row_number': 2, 'values': {'ProductName': 'Apples', 'Value': '68', 'Weight': '773'}}, {'data_row_number': 3, 'values': {'ProductName': 'Carrots', 'Value': '11', 'Weight': '653'}}, {'data_row_number': 4, 'values': {'ProductName': 'Basil', 'Value': '91', 'Weight': '755'}}, {'data_row_number': 5, 'values': {'ProductName': 'Potatoes', 'Value': '31', 'Weight': '670'}}, {'data_row_number': 6, 'values': {'ProductName': 'Green Beans', 'Value': '90', 'Weight': '505'}}, {'data_row_number': 7, 'values': {'ProductName': 'Blueberries', 'Value': '56', 'Weight': '821'}}, {'data_row_number': 8, 'values': {'ProductName': 'Oranges', 'Value': '10', 'Weight': '83'}}, {'data_row_number': 9, 'values': {'ProductName': 'Watermelons', 'Value': '24', 'Weight': '249'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    cap_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    if not cap_table['records']:
        raise ValueError('Capacity table has no records')
    cap_record = cap_table['records'][0]
    if 'Capacity' not in cap_record['values']:
        raise ValueError('Missing Capacity value')
    C = float(cap_record['values']['Capacity'])
    prod_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'))
    records = prod_table['records']
    indices = []
    w = {}
    v = {}
    for rec in records:
        idx = rec['data_row_number'] + 1
        indices.append(idx)
        vals = rec['values']
        if 'Weight' not in vals or 'Value' not in vals:
            raise ValueError(f"Missing Weight or Value for product row {rec['data_row_number']}")
        w[idx] = float(vals['Weight'])
        v[idx] = float(vals['Value'])
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(indices, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((v[i] * x[i] for i in indices)), GRB.MAXIMIZE)
    model.addConstr(gurobipy.quicksum((w[i] * x[i] for i in indices)) <= C, name='capacity')
    model.optimize()
    return model
m = solve_problem()
