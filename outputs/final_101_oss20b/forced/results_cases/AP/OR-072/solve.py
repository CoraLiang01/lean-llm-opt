CSVQA_DATA = {'query': 'For a certain daytime and nighttime service bus route, the number of drivers and crew members required during each time period is given in 42.csv:\n\n    Assuming that the drivers and crew members start working at the beginning of each time period and work continuously for 4 hours, how many drivers and crew members should be assigned to this bus route at least? Write down the linear programming model for this problem.', 'relationships': [], 'tables': [{'columns': ['Shift', 'Time', 'Number Required'], 'file_name': '42.csv', 'records': [{'data_row_number': 0, 'values': {'Number Required': '20', 'Shift': '1', 'Time': '0:00-1:00'}}, {'data_row_number': 1, 'values': {'Number Required': '18', 'Shift': '2', 'Time': '1:00-2:00'}}, {'data_row_number': 2, 'values': {'Number Required': '15', 'Shift': '3', 'Time': '2:00-3:00'}}, {'data_row_number': 3, 'values': {'Number Required': '15', 'Shift': '4', 'Time': '3:00-4:00'}}, {'data_row_number': 4, 'values': {'Number Required': '20', 'Shift': '5', 'Time': '4:00-5:00'}}, {'data_row_number': 5, 'values': {'Number Required': '30', 'Shift': '6', 'Time': '5:00-6:00'}}, {'data_row_number': 6, 'values': {'Number Required': '60', 'Shift': '7', 'Time': '6:00-7:00'}}, {'data_row_number': 7, 'values': {'Number Required': '70', 'Shift': '8', 'Time': '7:00-8:00'}}, {'data_row_number': 8, 'values': {'Number Required': '50', 'Shift': '9', 'Time': '8:00-9:00'}}, {'data_row_number': 9, 'values': {'Number Required': '55', 'Shift': '10', 'Time': '9:00-10:00'}}, {'data_row_number': 10, 'values': {'Number Required': '65', 'Shift': '11', 'Time': '10:00-11:00'}}, {'data_row_number': 11, 'values': {'Number Required': '75', 'Shift': '12', 'Time': '11:00-12:00'}}, {'data_row_number': 12, 'values': {'Number Required': '80', 'Shift': '13', 'Time': '12:00-13:00'}}, {'data_row_number': 13, 'values': {'Number Required': '70', 'Shift': '14', 'Time': '13:00-14:00'}}, {'data_row_number': 14, 'values': {'Number Required': '60', 'Shift': '15', 'Time': '14:00-15:00'}}, {'data_row_number': 15, 'values': {'Number Required': '55', 'Shift': '16', 'Time': '15:00-16:00'}}, {'data_row_number': 16, 'values': {'Number Required': '60', 'Shift': '17', 'Time': '16:00-17:00'}}, {'data_row_number': 17, 'values': {'Number Required': '75', 'Shift': '18', 'Time': '17:00-18:00'}}, {'data_row_number': 18, 'values': {'Number Required': '85', 'Shift': '19', 'Time': '18:00-19:00'}}, {'data_row_number': 19, 'values': {'Number Required': '70', 'Shift': '20', 'Time': '19:00-20:00'}}, {'data_row_number': 20, 'values': {'Number Required': '50', 'Shift': '21', 'Time': '20:00-21:00'}}, {'data_row_number': 21, 'values': {'Number Required': '40', 'Shift': '22', 'Time': '21:00-22:00'}}, {'data_row_number': 22, 'values': {'Number Required': '35', 'Shift': '23', 'Time': '22:00-23:00'}}, {'data_row_number': 23, 'values': {'Number Required': '25', 'Shift': '24', 'Time': '23:00-0:00'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    S = list(range(1, 25))
    r = {}
    for record in CSVQA_DATA['tables'][0]['records']:
        shift_str = record['values']['Shift']
        req_str = record['values']['Number Required']
        try:
            shift = int(shift_str)
            req = int(req_str)
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {record['data_row_number']}: {e}")
        r[shift] = req
    for i in S:
        if i not in r:
            raise ValueError(f'Missing required number for shift {i}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(S, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((x[i] for i in S)), GRB.MINIMIZE)
    for i in S:
        start = max(1, i - 3)
        model.addConstr(gurobipy.quicksum((x[s] for s in range(start, i + 1))) >= r[i], name=f'cover_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
