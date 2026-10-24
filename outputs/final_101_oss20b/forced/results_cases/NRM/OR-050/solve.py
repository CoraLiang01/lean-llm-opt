CSVQA_DATA = {'query': 'In retail, shops need to allocate various types of products to different displays. The capacity limit of each display is provided in "capacity.csv", and the value and weight of each product are provided in "products.csv". The objective is to determine the optimal number of each product to place on each display so as to maximize the total value of all products placed across the displays, while ensuring that the total weight of the products on each display does not exceed its capacity. In addition, the total quantity of the first product placed across all displays must be at least 5. The decision variable x_{ij} represents the number of units of product j placed on display i.', 'relationships': [], 'tables': [{'columns': ['ShelfID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '5.0', 'ShelfID': '1'}}, {'data_row_number': 1, 'values': {'Capacity': '7.0', 'ShelfID': '2'}}, {'data_row_number': 2, 'values': {'Capacity': '6.0', 'ShelfID': '3'}}, {'data_row_number': 3, 'values': {'Capacity': '8.0', 'ShelfID': '4'}}, {'data_row_number': 4, 'values': {'Capacity': '5.5', 'ShelfID': '5'}}, {'data_row_number': 5, 'values': {'Capacity': '9.0', 'ShelfID': '6'}}, {'data_row_number': 6, 'values': {'Capacity': '6.5', 'ShelfID': '7'}}, {'data_row_number': 7, 'values': {'Capacity': '7.5', 'ShelfID': '8'}}, {'data_row_number': 8, 'values': {'Capacity': '8.2', 'ShelfID': '9'}}, {'data_row_number': 9, 'values': {'Capacity': '5.7', 'ShelfID': '10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}, {'data_row_number': 1, 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}}, {'data_row_number': 2, 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'data_row_number': 3, 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}}, {'data_row_number': 4, 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'data_row_number': 5, 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'data_row_number': 6, 'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}}, {'data_row_number': 7, 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'data_row_number': 8, 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'data_row_number': 9, 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}}, {'data_row_number': 10, 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}}, {'data_row_number': 11, 'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'data_row_number': 12, 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'data_row_number': 13, 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'data_row_number': 14, 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'data_row_number': 15, 'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'data_row_number': 16, 'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'data_row_number': 17, 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}}, {'data_row_number': 18, 'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'data_row_number': 19, 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}], 'table_id': 'file_1_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}], 'table_id': 'file_1_view_1'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def find_table(table_id):
        for tbl in tables:
            if tbl.get('table_id') == table_id:
                return tbl
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    cap_table = find_table('file_0_view_0')
    capacities = {}
    for rec in cap_table['records']:
        vals = rec['values']
        if 'ShelfID' not in vals or 'Capacity' not in vals:
            raise ValueError('Missing ShelfID or Capacity in capacity record')
        shelf = vals['ShelfID']
        if shelf in capacities:
            raise ValueError(f'Duplicate ShelfID {shelf} in capacities')
        try:
            cap_val = float(vals['Capacity'])
        except Exception as e:
            raise ValueError(f'Non-numeric Capacity for ShelfID {shelf}') from e
        capacities[shelf] = cap_val
    prod_table = find_table('file_1_view_0')
    values = {}
    weights = {}
    for rec in prod_table['records']:
        vals = rec['values']
        if 'ProductName' not in vals or 'Value' not in vals or 'Weight' not in vals:
            raise ValueError('Missing ProductName, Value, or Weight in product record')
        prod = vals['ProductName']
        if prod in values or prod in weights:
            raise ValueError(f'Duplicate ProductName {prod} in products')
        try:
            val = float(vals['Value'])
            wt = float(vals['Weight'])
        except Exception as e:
            raise ValueError(f'Non-numeric Value or Weight for ProductName {prod}') from e
        values[prod] = val
        weights[prod] = wt
    if set(values.keys()) != set(weights.keys()):
        raise ValueError('Mismatch between product names in values and weights')
    jstar_table = find_table('file_1_view_1')
    j_star = None
    for rec in jstar_table['records']:
        vals = rec['values']
        if 'ProductName' not in vals:
            raise ValueError('Missing ProductName in first product record')
        j_star = vals['ProductName']
        break
    if j_star is None:
        raise ValueError('First product record not found')
    if j_star not in values or j_star not in weights:
        raise ValueError(f'First product {j_star} not present in product data')
    I = list(capacities.keys())
    J = list(values.keys())
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = model.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((values[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(gurobipy.quicksum((weights[j] * x[i, j] for j in J)) <= capacities[i], name=f'cap_{i}')
    model.addConstr(gurobipy.quicksum((x[i, j_star] for i in I)) >= 5, name='min_first')
    model.optimize()
    return model
m = solve_problem()
