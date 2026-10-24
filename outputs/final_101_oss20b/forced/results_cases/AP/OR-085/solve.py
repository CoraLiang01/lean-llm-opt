CSVQA_DATA = {'query': 'A traveling salesman must visit 15 customers at 15 different locations, and the (symmetric) distance matrix is restored in 20.csv file.\n    Develop a mathematical program to determine the visiting order starting from location 1 and ending at location 1, in order to minimize the traveling distance.', 'relationships': [], 'tables': [{'columns': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'], 'file_name': '20.csv', 'records': [{'data_row_number': 0, 'values': {'10': '28', '11': '55', '12': '53', '13': '66', '14': '89', '15': '78', '2': '67', '3': '55', '4': '80', '5': '21', '6': '77', '7': '78', '8': '74', '9': '85'}}, {'data_row_number': 1, 'values': {'10': '92', '11': '37', '12': '51', '13': '38', '14': '82', '15': '31', '2': '', '3': '38', '4': '29', '5': '68', '6': '36', '7': '62', '8': '54', '9': '49'}}, {'data_row_number': 2, 'values': {'10': '68', '11': '70', '12': '55', '13': '46', '14': '32', '15': '40', '2': '', '3': '', '4': '28', '5': '44', '6': '27', '7': '56', '8': '34', '9': '33'}}, {'data_row_number': 3, 'values': {'10': '55', '11': '68', '12': '85', '13': '58', '14': '56', '15': '22', '2': '', '3': '', '4': '', '5': '21', '6': '51', '7': '46', '8': '48', '9': '31'}}, {'data_row_number': 4, 'values': {'10': '79', '11': '49', '12': '70', '13': '43', '14': '55', '15': '78', '2': '', '3': '', '4': '', '5': '', '6': '42', '7': '57', '8': '31', '9': '55'}}, {'data_row_number': 5, 'values': {'10': '52', '11': '76', '12': '54', '13': '59', '14': '44', '15': '76', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '63', '8': '41', '9': '39'}}, {'data_row_number': 6, 'values': {'10': '37', '11': '55', '12': '54', '13': '51', '14': '14', '15': '64', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '38', '9': '35'}}, {'data_row_number': 7, 'values': {'10': '24', '11': '60', '12': '42', '13': '31', '14': '42', '15': '27', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': '53'}}, {'data_row_number': 8, 'values': {'10': '88', '11': '28', '12': '65', '13': '12', '14': '63', '15': '45', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': ''}}, {'data_row_number': 9, 'values': {'10': '', '11': '84', '12': '40', '13': '43', '14': '81', '15': '37', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': ''}}, {'data_row_number': 10, 'values': {'10': '', '11': '', '12': '41', '13': '38', '14': '56', '15': '35', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': ''}}, {'data_row_number': 11, 'values': {'10': '', '11': '', '12': '', '13': '65', '14': '47', '15': '38', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': ''}}, {'data_row_number': 12, 'values': {'10': '', '11': '', '12': '', '13': '', '14': '35', '15': '77', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': ''}}, {'data_row_number': 13, 'values': {'10': '', '11': '', '12': '', '13': '', '14': '', '15': '54', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': ''}}, {'data_row_number': 14, 'values': {'10': '', '11': '', '12': '', '13': '', '14': '', '15': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': '', '9': ''}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == table_id), None)
    if table is None:
        raise ValueError(f'Table with id {table_id} not found in CSVQA_DATA')
    records = table['records']
    N = list(range(1, 16))
    d = {}
    for i in N:
        for j in N:
            if i == j:
                continue
            if i < j:
                record = records[i - 1]
                val_str = record['values'].get(str(j), '')
                if val_str == '':
                    raise ValueError(f'Missing distance for pair ({i},{j})')
                try:
                    val = int(val_str)
                except ValueError:
                    raise ValueError(f'Invalid integer for distance ({i},{j}): {val_str}')
                d[i, j] = val
            else:
                d[i, j] = d[j, i]
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in N for j in N if i != j], vtype=GRB.BINARY, name='')
    u = m.addVars([i for i in N if i != 1], vtype=GRB.INTEGER, lb=2, ub=15, name='')
    m.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in N if j != i)) == 1 for i in N), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in N if i != j)) == 1 for j in N), name='')
    m.addConstrs((u[i] - u[j] + 15 * x[i, j] <= 14 for i in N for j in N if i != j and i != 1 and (j != 1)), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
