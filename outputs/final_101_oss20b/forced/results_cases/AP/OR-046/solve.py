CSVQA_DATA = {'query': 'A supermarket needs to replenish its stock, and the ‘products.csv’ provides the relevant income statement for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an overall stock capacity constraint detailed in ‘capacity.csv’.The objective is to decide which products to order each day and the quantities to be ordered in order to maximise the overall benefits while adhering to the overall stock capacity. The decision variable x_i represents the number of units of the ith product to be ordered each day.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '875'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Weight', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Spinach', 'Value': '64', 'Weight': '230'}}, {'data_row_number': 1, 'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '75', 'Weight': '637'}}, {'data_row_number': 2, 'values': {'ProductName': 'Apples', 'Value': '68', 'Weight': '773'}}, {'data_row_number': 3, 'values': {'ProductName': 'Carrots', 'Value': '11', 'Weight': '653'}}, {'data_row_number': 4, 'values': {'ProductName': 'Basil', 'Value': '91', 'Weight': '755'}}, {'data_row_number': 5, 'values': {'ProductName': 'Potatoes', 'Value': '31', 'Weight': '670'}}, {'data_row_number': 6, 'values': {'ProductName': 'Green Beans', 'Value': '90', 'Weight': '505'}}, {'data_row_number': 7, 'values': {'ProductName': 'Blueberries', 'Value': '56', 'Weight': '821'}}, {'data_row_number': 8, 'values': {'ProductName': 'Oranges', 'Value': '10', 'Weight': '83'}}, {'data_row_number': 9, 'values': {'ProductName': 'Watermelons', 'Value': '24', 'Weight': '249'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    if capacity_table is None or not capacity_table['records']:
        raise ValueError('Capacity table missing or empty')
    capacity_value = capacity_table['records'][0]['values'].get('Capacity')
    if capacity_value is None:
        raise ValueError('Capacity value missing')
    try:
        C = float(capacity_value)
    except ValueError:
        raise ValueError(f'Invalid capacity value: {capacity_value}')
    product_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'), None)
    if product_table is None or not product_table['records']:
        raise ValueError('Product table missing or empty')
    product_names = []
    weights = {}
    values = {}
    for rec in product_table['records']:
        vals = rec['values']
        name = vals.get('ProductName')
        if name is None:
            raise ValueError('ProductName missing in a record')
        weight_str = vals.get('Weight')
        value_str = vals.get('Value')
        if weight_str is None or value_str is None:
            raise ValueError(f'Weight or Value missing for product {name}')
        try:
            weight = float(weight_str)
            value = float(value_str)
        except ValueError:
            raise ValueError(f'Invalid numeric value for product {name}')
        product_names.append(name)
        weights[name] = weight
        values[name] = value
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((values[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    model.addConstr(gurobipy.quicksum((weights[i] * x[i] for i in product_names)) <= C, name='capacity')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
