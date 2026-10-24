CSVQA_DATA = {'query': 'In the context of BigMart Sales, the store needs to allocate various types of products into different display shelves. Specifically, the store has several shelves, each with a capacity limit provided in “capacity.csv.” The predefined value and weight of each product can be found in “products.csv.” The objective is to determine the optimal number of units of each product to place on each shelf to maximize the total value of the products across all shelves, while ensuring that the total weight of the products on each shelf does not exceed its capacity. The decision variables x_ij represent the number of units of product j to be placed on shelf i.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['ShelfID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '500', 'ShelfID': '1'}}, {'data_row_number': 1, 'values': {'Capacity': '700', 'ShelfID': '2'}}, {'data_row_number': 2, 'values': {'Capacity': '600', 'ShelfID': '3'}}, {'data_row_number': 3, 'values': {'Capacity': '800', 'ShelfID': '4'}}, {'data_row_number': 4, 'values': {'Capacity': '550', 'ShelfID': '5'}}, {'data_row_number': 5, 'values': {'Capacity': '900', 'ShelfID': '6'}}, {'data_row_number': 6, 'values': {'Capacity': '650', 'ShelfID': '7'}}, {'data_row_number': 7, 'values': {'Capacity': '750', 'ShelfID': '8'}}, {'data_row_number': 8, 'values': {'Capacity': '820', 'ShelfID': '9'}}, {'data_row_number': 9, 'values': {'Capacity': '570', 'ShelfID': '10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': '1', 'Value': '50', 'Weight': '10'}}, {'data_row_number': 1, 'values': {'ProductName': '2', 'Value': '70', 'Weight': '20'}}, {'data_row_number': 2, 'values': {'ProductName': '3', 'Value': '30', 'Weight': '5'}}, {'data_row_number': 3, 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'data_row_number': 4, 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'data_row_number': 5, 'values': {'ProductName': '6', 'Value': '90', 'Weight': '30'}}, {'data_row_number': 6, 'values': {'ProductName': '7', 'Value': '40', 'Weight': '12'}}, {'data_row_number': 7, 'values': {'ProductName': '8', 'Value': '100', 'Weight': '35'}}, {'data_row_number': 8, 'values': {'ProductName': '9', 'Value': '55', 'Weight': '10'}}, {'data_row_number': 9, 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'data_row_number': 10, 'values': {'ProductName': '11', 'Value': '65', 'Weight': '18'}}, {'data_row_number': 11, 'values': {'ProductName': '12', 'Value': '95', 'Weight': '28'}}, {'data_row_number': 12, 'values': {'ProductName': '13', 'Value': '45', 'Weight': '8'}}, {'data_row_number': 13, 'values': {'ProductName': '14', 'Value': '85', 'Weight': '22'}}, {'data_row_number': 14, 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'data_row_number': 15, 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'data_row_number': 16, 'values': {'ProductName': '17', 'Value': '50', 'Weight': '14'}}, {'data_row_number': 17, 'values': {'ProductName': '18', 'Value': '60', 'Weight': '16'}}, {'data_row_number': 18, 'values': {'ProductName': '19', 'Value': '120', 'Weight': '50'}}, {'data_row_number': 19, 'values': {'ProductName': '20', 'Value': '100', 'Weight': '30'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    shelves_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    if shelves_table is None:
        raise ValueError("Shelves table 'file_0_view_0' not found in CSVQA_DATA")
    products_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'), None)
    if products_table is None:
        raise ValueError("Products table 'file_1_view_0' not found in CSVQA_DATA")
    shelves = []
    capacity = {}
    for record in shelves_table['records']:
        values = record['values']
        if 'ShelfID' not in values:
            raise ValueError(f'Missing ShelfID in record {record}')
        if 'Capacity' not in values:
            raise ValueError(f'Missing Capacity in record {record}')
        shelf_id = values['ShelfID']
        shelves.append(shelf_id)
        try:
            capacity[shelf_id] = int(values['Capacity'])
        except ValueError:
            raise ValueError(f'Invalid Capacity value for shelf {shelf_id}')
    products = []
    value = {}
    weight = {}
    for record in products_table['records']:
        values = record['values']
        if 'ProductName' not in values:
            raise ValueError(f'Missing ProductName in record {record}')
        if 'Value' not in values:
            raise ValueError(f'Missing Value in record {record}')
        if 'Weight' not in values:
            raise ValueError(f'Missing Weight in record {record}')
        product_name = values['ProductName']
        products.append(product_name)
        try:
            value[product_name] = int(values['Value'])
        except ValueError:
            raise ValueError(f'Invalid Value for product {product_name}')
        try:
            weight[product_name] = int(values['Weight'])
        except ValueError:
            raise ValueError(f'Invalid Weight for product {product_name}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(s, p) for s in shelves for p in products], lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
    for s in shelves:
        m.addConstr(gurobipy.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s], name=f'cap_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
