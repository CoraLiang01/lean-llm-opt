CSVQA_DATA = {'query': 'A property developer is planning to develop real estate in New York.There are several areas to choose from, such as Queens and Brooklyn.However, due to limited resources and focus, the developer is not able to develop all the real estate in all the areas and has to select a few for development.The development benefit data for real estate in each area is recorded in the ‘products.csv’ file.The developer has an overall development capacity limit, which is detailed in the ‘capacity.csv’ file.The goal is to decide how many real estate in each areas are to be developed, in order to maximise the overall benefits while adhering to the overall development capacity.The decision variable x_i represents the scale of development per day in area i.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '586'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'data_row_number': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'data_row_number': 2, 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'data_row_number': 3, 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'data_row_number': 4, 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'data_row_number': 5, 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'data_row_number': 6, 'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'data_row_number': 7, 'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'data_row_number': 8, 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'data_row_number': 9, 'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'data_row_number': 10, 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'data_row_number': 11, 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'data_row_number': 12, 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'data_row_number': 13, 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'data_row_number': 14, 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'data_row_number': 15, 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'data_row_number': 16, 'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'data_row_number': 17, 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'data_row_number': 18, 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'data_row_number': 19, 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    if not capacity_table['records']:
        raise ValueError('Capacity record missing')
    capacity_value = capacity_table['records'][0]['values']['Capacity']
    try:
        C = float(capacity_value)
    except ValueError:
        raise ValueError(f'Invalid capacity value: {capacity_value}')
    products_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    if not products_table['records']:
        raise ValueError('Product records missing')
    I = []
    b_i = {}
    w_i = {}
    for rec in products_table['records']:
        vals = rec['values']
        prod = vals['ProductName']
        if prod in I:
            raise ValueError(f'Duplicate product name: {prod}')
        I.append(prod)
        try:
            b_i[prod] = float(vals['Value'])
        except ValueError:
            raise ValueError(f"Invalid benefit value for {prod}: {vals['Value']}")
        try:
            w_i[prod] = float(vals['Weight'])
        except ValueError:
            raise ValueError(f"Invalid weight value for {prod}: {vals['Weight']}")
    for prod in I:
        if prod not in b_i:
            raise ValueError(f'Missing benefit for product {prod}')
        if prod not in w_i:
            raise ValueError(f'Missing weight for product {prod}')
    m = gurobipy.Model('RealEstateDevelopment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((b_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w_i[i] * x[i] for i in I)) <= C)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
