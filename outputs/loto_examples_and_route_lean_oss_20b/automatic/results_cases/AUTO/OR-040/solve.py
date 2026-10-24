CSVQA_DATA = {'query': 'A real-estate developer is planning property development in New York City. For each area (e.g., Queens, Brooklyn, etc.), the developer has a “products.csv” file that records the benefit coefficient for that area.\n    The developer also faces a single overall development-capacity constraint, provided in “capacity.csv.” file. The objective is to decide the daily scale of development in each area so as to maximize total benefit while ensuring that the sum of all development units does not exceed the overall capacity. The decision variables  x_i  represent the scale of development in area  i  each day.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '4466'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Queens', 'Value': '443', 'Weight': '104'}}, {'data_row_number': 1, 'values': {'ProductName': 'Brooklyn', 'Value': '522', 'Weight': '368'}}, {'data_row_number': 2, 'values': {'ProductName': 'Manhattan', 'Value': '300', 'Weight': '483'}}, {'data_row_number': 3, 'values': {'ProductName': 'Bronx', 'Value': '767', 'Weight': '165'}}, {'data_row_number': 4, 'values': {'ProductName': 'Staten Island', 'Value': '300', 'Weight': '105'}}, {'data_row_number': 5, 'values': {'ProductName': 'Harlem', 'Value': '309', 'Weight': '123'}}, {'data_row_number': 6, 'values': {'ProductName': 'Upper East Side', 'Value': '598', 'Weight': '131'}}, {'data_row_number': 7, 'values': {'ProductName': 'Lower Manhattan', 'Value': '460', 'Weight': '341'}}, {'data_row_number': 8, 'values': {'ProductName': 'Midtown', 'Value': '318', 'Weight': '258'}}, {'data_row_number': 9, 'values': {'ProductName': 'Long Island City', 'Value': '126', 'Weight': '469'}}, {'data_row_number': 10, 'values': {'ProductName': 'Williamsburg', 'Value': '593', 'Weight': '387'}}, {'data_row_number': 11, 'values': {'ProductName': 'Bushwick', 'Value': '871', 'Weight': '425'}}, {'data_row_number': 12, 'values': {'ProductName': 'Flatbush', 'Value': '858', 'Weight': '482'}}, {'data_row_number': 13, 'values': {'ProductName': 'Greenpoint', 'Value': '321', 'Weight': '495'}}, {'data_row_number': 14, 'values': {'ProductName': 'Park Slope', 'Value': '275', 'Weight': '305'}}, {'data_row_number': 15, 'values': {'ProductName': 'Astoria', 'Value': '700', 'Weight': '377'}}, {'data_row_number': 16, 'values': {'ProductName': 'Jackson Heights', 'Value': '685', 'Weight': '318'}}, {'data_row_number': 17, 'values': {'ProductName': 'Flushing', 'Value': '940', 'Weight': '56'}}, {'data_row_number': 18, 'values': {'ProductName': 'Sunnyside', 'Value': '522', 'Weight': '213'}}, {'data_row_number': 19, 'values': {'ProductName': 'Ditmars', 'Value': '763', 'Weight': '472'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    cap_table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if cap_table is None or not cap_table.get('records'):
        raise ValueError("Capacity table 'file_0_view_0' missing or empty")
    cap_record = cap_table['records'][0]
    try:
        C = int(cap_record['values']['Capacity'])
    except (KeyError, ValueError) as e:
        raise ValueError(f'Invalid capacity value: {e}')
    prod_table = next((t for t in tables if t.get('table_id') == 'file_1_view_0'), None)
    if prod_table is None or not prod_table.get('records'):
        raise ValueError("Products table 'file_1_view_0' missing or empty")
    records = prod_table['records']
    I = []
    v_i = {}
    for rec in records:
        try:
            pname = rec['values']['ProductName']
            val = int(rec['values']['Value'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Invalid product record: {e}')
        if pname in v_i:
            raise ValueError(f'Duplicate product name detected: {pname}')
        I.append(pname)
        v_i[pname] = val
    m = gurobipy.Model('RealEstateDevelopment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((v_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.Status}')
