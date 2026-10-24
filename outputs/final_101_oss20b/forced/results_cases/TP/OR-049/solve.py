CSVQA_DATA = {'query': 'In the context of retail sales, the store needs to allocate various types of products into different display shelves. Specifically, the store has several display shelves, each with a capacity limit provided in “capacity.csv.” The predefined value and weight of each product can be found in “products.csv.” The objective is to determine the optimal number of units of each product to place on each display shelf to maximize the total value of the products across all shelves while ensuring that the total weight of the products on each shelf does not exceed its capacity. The decision variablesx_ijrepresent the number of units of product j to be placed on shelf i.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['ShelfID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '5.0', 'ShelfID': '1'}}, {'data_row_number': 1, 'values': {'Capacity': '7.0', 'ShelfID': '2'}}, {'data_row_number': 2, 'values': {'Capacity': '6.0', 'ShelfID': '3'}}, {'data_row_number': 3, 'values': {'Capacity': '8.0', 'ShelfID': '4'}}, {'data_row_number': 4, 'values': {'Capacity': '5.5', 'ShelfID': '5'}}, {'data_row_number': 5, 'values': {'Capacity': '9.0', 'ShelfID': '6'}}, {'data_row_number': 6, 'values': {'Capacity': '6.5', 'ShelfID': '7'}}, {'data_row_number': 7, 'values': {'Capacity': '7.5', 'ShelfID': '8'}}, {'data_row_number': 8, 'values': {'Capacity': '8.2', 'ShelfID': '9'}}, {'data_row_number': 9, 'values': {'Capacity': '5.7', 'ShelfID': '10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}, {'data_row_number': 1, 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}}, {'data_row_number': 2, 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'data_row_number': 3, 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}}, {'data_row_number': 4, 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'data_row_number': 5, 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'data_row_number': 6, 'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}}, {'data_row_number': 7, 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'data_row_number': 8, 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'data_row_number': 9, 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}}, {'data_row_number': 10, 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}}, {'data_row_number': 11, 'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'data_row_number': 12, 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'data_row_number': 13, 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'data_row_number': 14, 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'data_row_number': 15, 'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'data_row_number': 16, 'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'data_row_number': 17, 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}}, {'data_row_number': 18, 'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'data_row_number': 19, 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    shelf_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    product_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    I = [rec['values']['ShelfID'] for rec in shelf_table['records']]
    J = [rec['values']['ProductName'] for rec in product_table['records']]
    cap = {}
    for rec in shelf_table['records']:
        key = rec['values']['ShelfID']
        if key in cap:
            raise ValueError(f'Duplicate capacity entry for shelf {key}')
        try:
            cap[key] = float(rec['values']['Capacity'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid capacity for shelf {key}') from e
    val = {}
    wt = {}
    for rec in product_table['records']:
        key = rec['values']['ProductName']
        if key in val or key in wt:
            raise ValueError(f'Duplicate product entry for {key}')
        try:
            val[key] = float(rec['values']['Value'])
            wt[key] = float(rec['values']['Weight'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid value/weight for product {key}') from e
    for i in I:
        if i not in cap:
            raise ValueError(f'Missing capacity for shelf {i}')
    for j in J:
        if j not in val or j not in wt:
            raise ValueError(f'Missing value or weight for product {j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((val[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((wt[j] * x[i, j] for j in J)) <= cap[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
