CSVQA_DATA = {'query': 'A supermarket needs to restock its inventory, and for each type of produce (such as leafy vegetables, mushrooms, etc.), there is an associated benefit table provided in "products.csv." The supermarket faces an overall inventory-capacity constraint, provided in “capacity.csv.”. The goal is to decide the daily order quantity of each produce type so as to maximize total benefit while ensuring that the total weight of all ordered units does not exceed the overall capacity.The decision variables x_i represents the number of units of type i of produce to be ordered daily.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '1035'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Weight', 'Value'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Spinach', 'Value': '49', 'Weight': '282'}}, {'data_row_number': 1, 'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '30', 'Weight': '83'}}, {'data_row_number': 2, 'values': {'ProductName': 'Apples', 'Value': '30', 'Weight': '251'}}, {'data_row_number': 3, 'values': {'ProductName': 'Carrots', 'Value': '18', 'Weight': '257'}}, {'data_row_number': 4, 'values': {'ProductName': 'Basil', 'Value': '54', 'Weight': '88'}}, {'data_row_number': 5, 'values': {'ProductName': 'Potatoes', 'Value': '27', 'Weight': '52'}}, {'data_row_number': 6, 'values': {'ProductName': 'Green Beans', 'Value': '91', 'Weight': '198'}}, {'data_row_number': 7, 'values': {'ProductName': 'Blueberries', 'Value': '88', 'Weight': '203'}}, {'data_row_number': 8, 'values': {'ProductName': 'Oranges', 'Value': '78', 'Weight': '87'}}, {'data_row_number': 9, 'values': {'ProductName': 'Watermelons', 'Value': '22', 'Weight': '265'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    if not cap_table['records']:
        raise ValueError('Capacity record missing')
    cap_value_str = cap_table['records'][0]['values']['Capacity']
    try:
        C = int(cap_value_str)
    except ValueError:
        raise ValueError('Capacity value is not an integer')
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    if not prod_table['records']:
        raise ValueError('Product records missing')
    product_names = []
    weights = {}
    values = {}
    for rec in prod_table['records']:
        vals = rec['values']
        name = vals['ProductName']
        if name in product_names:
            raise ValueError(f'Duplicate product name {name}')
        product_names.append(name)
        try:
            w = int(vals['Weight'])
        except ValueError:
            raise ValueError(f'Weight for {name} is not an integer')
        try:
            v = int(vals['Value'])
        except ValueError:
            raise ValueError(f'Value for {name} is not an integer')
        weights[name] = w
        values[name] = v
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((values[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weights[i] * x[i] for i in product_names)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
