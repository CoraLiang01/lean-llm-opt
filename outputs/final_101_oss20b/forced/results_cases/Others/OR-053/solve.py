CSVQA_DATA = {'query': 'In the context of BigMart Sales, the store needs to allocate various types of products into different display shelves. Specifically, the store has several shelves, each with a capacity limit provided in “capacity.csv.” The predefined value and weight of each product can be found in “products.csv.” The objective is to determine the optimal number of units of each product to place on each shelf to maximize the total value of the products across all shelves, while ensuring that the total weight of the products on each shelf does not exceed its capacity. The decision variables x_ij represent the number of units of product j to be placed on shelf i.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['ShelfID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '500', 'ShelfID': '1'}}, {'data_row_number': 1, 'values': {'Capacity': '700', 'ShelfID': '2'}}, {'data_row_number': 2, 'values': {'Capacity': '600', 'ShelfID': '3'}}, {'data_row_number': 3, 'values': {'Capacity': '800', 'ShelfID': '4'}}, {'data_row_number': 4, 'values': {'Capacity': '550', 'ShelfID': '5'}}, {'data_row_number': 5, 'values': {'Capacity': '900', 'ShelfID': '6'}}, {'data_row_number': 6, 'values': {'Capacity': '650', 'ShelfID': '7'}}, {'data_row_number': 7, 'values': {'Capacity': '750', 'ShelfID': '8'}}, {'data_row_number': 8, 'values': {'Capacity': '820', 'ShelfID': '9'}}, {'data_row_number': 9, 'values': {'Capacity': '570', 'ShelfID': '10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': '1', 'Value': '50', 'Weight': '10'}}, {'data_row_number': 1, 'values': {'ProductName': '2', 'Value': '70', 'Weight': '20'}}, {'data_row_number': 2, 'values': {'ProductName': '3', 'Value': '30', 'Weight': '5'}}, {'data_row_number': 3, 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'data_row_number': 4, 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'data_row_number': 5, 'values': {'ProductName': '6', 'Value': '90', 'Weight': '30'}}, {'data_row_number': 6, 'values': {'ProductName': '7', 'Value': '40', 'Weight': '12'}}, {'data_row_number': 7, 'values': {'ProductName': '8', 'Value': '100', 'Weight': '35'}}, {'data_row_number': 8, 'values': {'ProductName': '9', 'Value': '55', 'Weight': '10'}}, {'data_row_number': 9, 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'data_row_number': 10, 'values': {'ProductName': '11', 'Value': '65', 'Weight': '18'}}, {'data_row_number': 11, 'values': {'ProductName': '12', 'Value': '95', 'Weight': '28'}}, {'data_row_number': 12, 'values': {'ProductName': '13', 'Value': '45', 'Weight': '8'}}, {'data_row_number': 13, 'values': {'ProductName': '14', 'Value': '85', 'Weight': '22'}}, {'data_row_number': 14, 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'data_row_number': 15, 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'data_row_number': 16, 'values': {'ProductName': '17', 'Value': '50', 'Weight': '14'}}, {'data_row_number': 17, 'values': {'ProductName': '18', 'Value': '60', 'Weight': '16'}}, {'data_row_number': 18, 'values': {'ProductName': '19', 'Value': '120', 'Weight': '50'}}, {'data_row_number': 19, 'values': {'ProductName': '20', 'Value': '100', 'Weight': '30'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model('ShelfProductAllocation')
    m.setParam('MIPGap', 1e-06)
    shelves_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    products_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'))
    I = []
    C_i = {}
    for rec in shelves_table['records']:
        sid = rec['values']['ShelfID']
        cap = int(rec['values']['Capacity'])
        I.append(sid)
        C_i[sid] = cap
    J = []
    w_j = {}
    v_j = {}
    for rec in products_table['records']:
        pname = rec['values']['ProductName']
        val = int(rec['values']['Value'])
        wt = int(rec['values']['Weight'])
        J.append(pname)
        v_j[pname] = val
        w_j[pname] = wt
    for i in I:
        if i not in C_i:
            raise ValueError(f'Missing capacity for shelf {i}')
    for j in J:
        if j not in w_j or j not in v_j:
            raise ValueError(f'Missing weight or value for product {j}')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((v_j[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((w_j[j] * x[i, j] for j in J)) <= C_i[i], name=f'cap_{i}')
    m.optimize()
    return m
m = solve_problem()
