CSVQA_DATA = {'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered each day.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '875'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Weight', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Spinach', 'Value': '64', 'Weight': '230'}}, {'data_row_number': 1, 'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '75', 'Weight': '637'}}, {'data_row_number': 2, 'values': {'ProductName': 'Apples', 'Value': '68', 'Weight': '773'}}, {'data_row_number': 3, 'values': {'ProductName': 'Carrots', 'Value': '11', 'Weight': '653'}}, {'data_row_number': 4, 'values': {'ProductName': 'Basil', 'Value': '91', 'Weight': '755'}}, {'data_row_number': 5, 'values': {'ProductName': 'Potatoes', 'Value': '31', 'Weight': '670'}}, {'data_row_number': 6, 'values': {'ProductName': 'Green Beans', 'Value': '90', 'Weight': '505'}}, {'data_row_number': 7, 'values': {'ProductName': 'Blueberries', 'Value': '56', 'Weight': '821'}}, {'data_row_number': 8, 'values': {'ProductName': 'Oranges', 'Value': '10', 'Weight': '83'}}, {'data_row_number': 9, 'values': {'ProductName': 'Watermelons', 'Value': '24', 'Weight': '249'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    cap_record = cap_table['records'][0]
    C = float(cap_record['values']['Capacity'])
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    prod_records = prod_table['records']
    ids = list(range(1, len(prod_records) + 1))
    w = {}
    v = {}
    for idx, rec in enumerate(prod_records, start=1):
        w[idx] = float(rec['values']['Weight'])
        v[idx] = float(rec['values']['Value'])
    if set(ids) != set(w.keys()) or set(ids) != set(v.keys()):
        raise ValueError('Missing parameters for some products')
    x = m.addVars(ids, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((v[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w[i] * x[i] for i in ids)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.Status)
