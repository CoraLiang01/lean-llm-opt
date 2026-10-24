CSVQA_DATA = {'query': 'In BigMart sales, the shop needs to allocate various types of products to different display shelves. The capacity of each shelf is provided in “capacity.csv,” while each product’s value and weight are in “products.csv.”The objective is to maximize the total value of products on all shelves without exceeding any shelf’s capacity. The decision variable  x_{ij} denotes how many units of product 𝑗 are placed on shelf 𝑖.', 'relationships': [], 'tables': [{'columns': ['ShelfID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '750', 'ShelfID': '1'}}, {'data_row_number': 1, 'values': {'Capacity': '820', 'ShelfID': '2'}}, {'data_row_number': 2, 'values': {'Capacity': '570', 'ShelfID': '3'}}, {'data_row_number': 3, 'values': {'Capacity': '800', 'ShelfID': '4'}}, {'data_row_number': 4, 'values': {'Capacity': '550', 'ShelfID': '5'}}, {'data_row_number': 5, 'values': {'Capacity': '900', 'ShelfID': '6'}}, {'data_row_number': 6, 'values': {'Capacity': '650', 'ShelfID': '7'}}, {'data_row_number': 7, 'values': {'Capacity': '800', 'ShelfID': '8'}}, {'data_row_number': 8, 'values': {'Capacity': '850', 'ShelfID': '9'}}, {'data_row_number': 9, 'values': {'Capacity': '900', 'ShelfID': '10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': '1', 'Value': '55', 'Weight': '10'}}, {'data_row_number': 1, 'values': {'ProductName': '2', 'Value': '75', 'Weight': '20'}}, {'data_row_number': 2, 'values': {'ProductName': '3', 'Value': '65', 'Weight': '5'}}, {'data_row_number': 3, 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'data_row_number': 4, 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'data_row_number': 5, 'values': {'ProductName': '6', 'Value': '90', 'Weight': '35'}}, {'data_row_number': 6, 'values': {'ProductName': '7', 'Value': '40', 'Weight': '45'}}, {'data_row_number': 7, 'values': {'ProductName': '8', 'Value': '100', 'Weight': '55'}}, {'data_row_number': 8, 'values': {'ProductName': '9', 'Value': '55', 'Weight': '65'}}, {'data_row_number': 9, 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'data_row_number': 10, 'values': {'ProductName': '11', 'Value': '110', 'Weight': '18'}}, {'data_row_number': 11, 'values': {'ProductName': '12', 'Value': '50', 'Weight': '28'}}, {'data_row_number': 12, 'values': {'ProductName': '13', 'Value': '60', 'Weight': '8'}}, {'data_row_number': 13, 'values': {'ProductName': '14', 'Value': '120', 'Weight': '28'}}, {'data_row_number': 14, 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'data_row_number': 15, 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'data_row_number': 16, 'values': {'ProductName': '17', 'Value': '50', 'Weight': '55'}}, {'data_row_number': 17, 'values': {'ProductName': '18', 'Value': '60', 'Weight': '70'}}, {'data_row_number': 18, 'values': {'ProductName': '19', 'Value': '120', 'Weight': '85'}}, {'data_row_number': 19, 'values': {'ProductName': '20', 'Value': '100', 'Weight': '100'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    shelf_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    product_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    I = []
    C_i = {}
    for rec in shelf_table['records']:
        sid = rec['values']['ShelfID']
        cap = int(rec['values']['Capacity'])
        I.append(sid)
        C_i[sid] = cap
    J = []
    w_j = {}
    v_j = {}
    for rec in product_table['records']:
        pid = rec['values']['ProductName']
        val = int(rec['values']['Value'])
        wt = int(rec['values']['Weight'])
        J.append(pid)
        v_j[pid] = val
        w_j[pid] = wt
    if set(I) != set(C_i.keys()):
        missing = set(I) - set(C_i.keys())
        raise ValueError(f'Missing capacity for shelves: {missing}')
    if set(J) != set(w_j.keys()) or set(J) != set(v_j.keys()):
        missing_w = set(J) - set(w_j.keys())
        missing_v = set(J) - set(v_j.keys())
        raise ValueError(f'Missing weight for products: {missing_w}, missing value for products: {missing_v}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, vtype=GRB.INTEGER, lb=0, name='')
    obj = gurobipy.quicksum((v_j[j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MAXIMIZE)
    for i in I:
        expr = gurobipy.quicksum((w_j[j] * x[i, j] for j in J))
        m.addConstr(expr <= C_i[i], name=f'cap_{i}')
    m.optimize()
    return m
m = solve_problem()
