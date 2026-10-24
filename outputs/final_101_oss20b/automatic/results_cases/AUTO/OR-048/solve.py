CSVQA_DATA = {'query': 'Amazon needs to allocate different types of air conditioners into different warehouse storage areas. Specifically, Amazon has several storage areas, each with a capacity limit provided in “capacity.csv.” The predefined value and size of each air conditioner type can be found in “products.csv.” The objective is to determine the optimal number of units of each air conditioner type to place in each storage area to maximize the total value of the air conditioners across all areas, while ensuring that the total size of the units in each area does not exceed its capacity. The decision variablesx_ijrepresent the number of units of air conditioner type j to be placed in storage area i.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['StorageID', 'Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '1083', 'StorageID': '1'}}, {'data_row_number': 1, 'values': {'Capacity': '1840', 'StorageID': '2'}}, {'data_row_number': 2, 'values': {'Capacity': '770', 'StorageID': '3'}}, {'data_row_number': 3, 'values': {'Capacity': '1299', 'StorageID': '4'}}, {'data_row_number': 4, 'values': {'Capacity': '1259', 'StorageID': '5'}}, {'data_row_number': 5, 'values': {'Capacity': '543', 'StorageID': '6'}}, {'data_row_number': 6, 'values': {'Capacity': '1831', 'StorageID': '7'}}, {'data_row_number': 7, 'values': {'Capacity': '855', 'StorageID': '8'}}, {'data_row_number': 8, 'values': {'Capacity': '619', 'StorageID': '9'}}, {'data_row_number': 9, 'values': {'Capacity': '637', 'StorageID': '10'}}, {'data_row_number': 10, 'values': {'Capacity': '935', 'StorageID': '11'}}, {'data_row_number': 11, 'values': {'Capacity': '626', 'StorageID': '12'}}, {'data_row_number': 12, 'values': {'Capacity': '1457', 'StorageID': '13'}}, {'data_row_number': 13, 'values': {'Capacity': '1198', 'StorageID': '14'}}, {'data_row_number': 14, 'values': {'Capacity': '837', 'StorageID': '15'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'Window Unit', 'Value': '4811', 'Weight': '114'}}, {'data_row_number': 1, 'values': {'ProductName': 'Portable Unit', 'Value': '1130', 'Weight': '200'}}, {'data_row_number': 2, 'values': {'ProductName': 'Split System', 'Value': '1611', 'Weight': '106'}}, {'data_row_number': 3, 'values': {'ProductName': 'Ductless System', 'Value': '3368', 'Weight': '256'}}, {'data_row_number': 4, 'values': {'ProductName': 'Central AC', 'Value': '2135', 'Weight': '268'}}, {'data_row_number': 5, 'values': {'ProductName': 'Hybrid AC', 'Value': '1046', 'Weight': '185'}}, {'data_row_number': 6, 'values': {'ProductName': 'Geothermal AC', 'Value': '4030', 'Weight': '299'}}, {'data_row_number': 7, 'values': {'ProductName': 'Smart AC', 'Value': '3761', 'Weight': '131'}}, {'data_row_number': 8, 'values': {'ProductName': 'Evaporative Cooler', 'Value': '3523', 'Weight': '139'}}, {'data_row_number': 9, 'values': {'ProductName': 'Package Unit', 'Value': '1701', 'Weight': '105'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    capacity = {}
    for rec in capacity_table['records']:
        sid = rec['values']['StorageID']
        cap = int(rec['values']['Capacity'])
        capacity[sid] = cap
    products_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    value = {}
    weight = {}
    for rec in products_table['records']:
        pname = rec['values']['ProductName']
        val = int(rec['values']['Value'])
        wt = int(rec['values']['Weight'])
        value[pname] = val
        weight[pname] = wt
    I = list(capacity.keys())
    J = list(value.keys())
    for pname in J:
        if pname not in weight:
            raise ValueError(f'Missing weight for product {pname}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((value[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    model.addConstrs((gurobipy.quicksum((weight[j] * x[i, j] for j in J)) <= capacity[i] for i in I), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
