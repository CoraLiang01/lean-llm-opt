CSVQA_DATA = {'query': 'A supermarket needs to restock its inventory, and for each type of produce (such as leafy vegetables, mushrooms, etc.), there is an associated benefit table provided in "products.csv." The supermarket faces an overall inventory-capacity constraint, provided in “capacity.csv.”. The goal is to decide the daily order quantity of each produce type so as to maximize total benefit while ensuring that the total weight of all ordered units does not exceed the overall capacity.The decision variables x_i represents the number of units of type i of produce to be ordered daily.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '1035'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Weight', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Spinach', 'Value': '49', 'Weight': '282'}}, {'data_row_number': 1, 'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '30', 'Weight': '83'}}, {'data_row_number': 2, 'values': {'ProductName': 'Apples', 'Value': '30', 'Weight': '251'}}, {'data_row_number': 3, 'values': {'ProductName': 'Carrots', 'Value': '18', 'Weight': '257'}}, {'data_row_number': 4, 'values': {'ProductName': 'Basil', 'Value': '54', 'Weight': '88'}}, {'data_row_number': 5, 'values': {'ProductName': 'Potatoes', 'Value': '27', 'Weight': '52'}}, {'data_row_number': 6, 'values': {'ProductName': 'Green Beans', 'Value': '91', 'Weight': '198'}}, {'data_row_number': 7, 'values': {'ProductName': 'Blueberries', 'Value': '88', 'Weight': '203'}}, {'data_row_number': 8, 'values': {'ProductName': 'Oranges', 'Value': '78', 'Weight': '87'}}, {'data_row_number': 9, 'values': {'ProductName': 'Watermelons', 'Value': '22', 'Weight': '265'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    if not cap_table['records']:
        raise ValueError('Capacity table has no records')
    cap_value_str = cap_table['records'][0]['values']['Capacity']
    try:
        C = int(cap_value_str)
    except ValueError:
        C = float(cap_value_str)
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    I = []
    w = {}
    v = {}
    for rec in prod_table['records']:
        prod_name = rec['values']['ProductName']
        weight_str = rec['values']['Weight']
        value_str = rec['values']['Value']
        try:
            weight = int(weight_str)
        except ValueError:
            weight = float(weight_str)
        try:
            value = int(value_str)
        except ValueError:
            value = float(value_str)
        if prod_name in I:
            raise ValueError(f'Duplicate product name {prod_name}')
        I.append(prod_name)
        w[prod_name] = weight
        v[prod_name] = value
    for prod in I:
        if prod not in w or prod not in v:
            raise ValueError(f'Missing weight or value for product {prod}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstr(gurobipy.quicksum((w[i] * x[i] for i in I)) <= C)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
