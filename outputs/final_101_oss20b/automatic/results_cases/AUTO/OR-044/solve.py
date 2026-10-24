CSVQA_DATA = {'query': 'A supermarket manager needs to select a variety of products to stock in different sections of the store. Particularly, the store has several sections, each with a display space limit provided in "capacity.csv." The predefined price and shelf space requirement of each product are detailed in "products.csv." The objective is to determine the optimal number of units of each product to stock in each section to maximize the total revenue, while ensuring that the total space used by the products in each section does not exceed the available capacity. The decision variables x_ij denote the number of units of product j to be placed in section i.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['SectionID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '100', 'SectionID': '1'}}, {'data_row_number': 1, 'values': {'Capacity': '150', 'SectionID': '2'}}, {'data_row_number': 2, 'values': {'Capacity': '120', 'SectionID': '3'}}, {'data_row_number': 3, 'values': {'Capacity': '130', 'SectionID': '4'}}, {'data_row_number': 4, 'values': {'Capacity': '90', 'SectionID': '5'}}, {'data_row_number': 5, 'values': {'Capacity': '110', 'SectionID': '6'}}, {'data_row_number': 6, 'values': {'Capacity': '160', 'SectionID': '7'}}, {'data_row_number': 7, 'values': {'Capacity': '140', 'SectionID': '8'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': '1', 'Value': '10', 'Weight': '2'}}, {'data_row_number': 1, 'values': {'ProductName': '2', 'Value': '15', 'Weight': '3'}}, {'data_row_number': 2, 'values': {'ProductName': '3', 'Value': '8', 'Weight': '1'}}, {'data_row_number': 3, 'values': {'ProductName': '4', 'Value': '12', 'Weight': '2'}}, {'data_row_number': 4, 'values': {'ProductName': '5', 'Value': '20', 'Weight': '4'}}, {'data_row_number': 5, 'values': {'ProductName': '6', 'Value': '25', 'Weight': '5'}}, {'data_row_number': 6, 'values': {'ProductName': '7', 'Value': '5', 'Weight': '1'}}, {'data_row_number': 7, 'values': {'ProductName': '8', 'Value': '30', 'Weight': '6'}}, {'data_row_number': 8, 'values': {'ProductName': '9', 'Value': '18', 'Weight': '3'}}, {'data_row_number': 9, 'values': {'ProductName': '10', 'Value': '22', 'Weight': '4'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = None
    product_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            capacity_table = tbl
        elif tbl.get('table_id') == 'file_1_view_0':
            product_table = tbl
    if capacity_table is None or product_table is None:
        raise ValueError('Required tables not found in CSVQA_DATA')
    I = [rec['values']['SectionID'] for rec in capacity_table['records']]
    J = [rec['values']['ProductName'] for rec in product_table['records']]
    capacity_dict = {}
    for rec in capacity_table['records']:
        key = rec['values']['SectionID']
        val = rec['values']['Capacity']
        capacity_dict[key] = int(val)
    value_dict = {}
    weight_dict = {}
    for rec in product_table['records']:
        key = rec['values']['ProductName']
        value_dict[key] = int(rec['values']['Value'])
        weight_dict[key] = int(rec['values']['Weight'])
    if not set(I).issubset(capacity_dict):
        missing = set(I) - set(capacity_dict)
        raise ValueError(f'Missing capacity for sections: {missing}')
    if not set(J).issubset(value_dict):
        missing = set(J) - set(value_dict)
        raise ValueError(f'Missing value for products: {missing}')
    if not set(J).issubset(weight_dict):
        missing = set(J) - set(weight_dict)
        raise ValueError(f'Missing weight for products: {missing}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value_dict[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((weight_dict[j] * x[i, j] for j in J)) <= capacity_dict[i], name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
