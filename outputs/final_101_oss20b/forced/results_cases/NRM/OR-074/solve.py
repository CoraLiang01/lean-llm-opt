CSVQA_DATA = {'query': 'A certain restaurant operates 24 hours a day and requires a minimum number of waitstaff as shown in 44.csv.\n\n    Each waitstaff works continuously for 8 hours a day. The goal is to find the minimum number of waitstaff that satisfies the above conditions and represent this problem as a linear programming model.', 'relationships': [], 'tables': [{'columns': ['Time', 'Requirement'], 'file_name': '44.csv', 'records': [{'data_row_number': 0, 'values': {'Requirement': '2', 'Time': '2:00am - 2:30am'}}, {'data_row_number': 1, 'values': {'Requirement': '3', 'Time': '2:30am - 3:00am'}}, {'data_row_number': 2, 'values': {'Requirement': '4', 'Time': '3:00am - 3:30am'}}, {'data_row_number': 3, 'values': {'Requirement': '6', 'Time': '3:30am - 4:00am'}}, {'data_row_number': 4, 'values': {'Requirement': '5', 'Time': '4:00am - 4:30am'}}, {'data_row_number': 5, 'values': {'Requirement': '4', 'Time': '4:30am - 5:00am'}}, {'data_row_number': 6, 'values': {'Requirement': '5', 'Time': '5:00am - 5:30am'}}, {'data_row_number': 7, 'values': {'Requirement': '6', 'Time': '5:30am - 6:00am'}}, {'data_row_number': 8, 'values': {'Requirement': '7', 'Time': '6:00am - 6:30am'}}, {'data_row_number': 9, 'values': {'Requirement': '8', 'Time': '6:30am - 7:00am'}}, {'data_row_number': 10, 'values': {'Requirement': '9', 'Time': '7:00am - 7:30am'}}, {'data_row_number': 11, 'values': {'Requirement': '9', 'Time': '7:30am - 8:00am'}}, {'data_row_number': 12, 'values': {'Requirement': '8', 'Time': '8:00am - 8:30am'}}, {'data_row_number': 13, 'values': {'Requirement': '8', 'Time': '8:30am - 9:00am'}}, {'data_row_number': 14, 'values': {'Requirement': '9', 'Time': '9:00am - 9:30am'}}, {'data_row_number': 15, 'values': {'Requirement': '9', 'Time': '9:30am - 10:00am'}}, {'data_row_number': 16, 'values': {'Requirement': '10', 'Time': '10:00am - 10:30am'}}, {'data_row_number': 17, 'values': {'Requirement': '12', 'Time': '10:30am - 11:00am'}}, {'data_row_number': 18, 'values': {'Requirement': '11', 'Time': '11:00am - 11:30am'}}, {'data_row_number': 19, 'values': {'Requirement': '11', 'Time': '11:30am - 12:00pm'}}, {'data_row_number': 20, 'values': {'Requirement': '12', 'Time': '12:00pm - 12:30pm'}}, {'data_row_number': 21, 'values': {'Requirement': '11', 'Time': '12:30pm - 1:00pm'}}, {'data_row_number': 22, 'values': {'Requirement': '10', 'Time': '1:00pm - 1:30pm'}}, {'data_row_number': 23, 'values': {'Requirement': '9', 'Time': '1:30pm - 2:00pm'}}, {'data_row_number': 24, 'values': {'Requirement': '8', 'Time': '2:00pm - 2:30pm'}}, {'data_row_number': 25, 'values': {'Requirement': '7', 'Time': '2:30pm - 3:00pm'}}, {'data_row_number': 26, 'values': {'Requirement': '6', 'Time': '3:00pm - 3:30pm'}}, {'data_row_number': 27, 'values': {'Requirement': '5', 'Time': '3:30pm - 4:00pm'}}, {'data_row_number': 28, 'values': {'Requirement': '5', 'Time': '4:00pm - 4:30pm'}}, {'data_row_number': 29, 'values': {'Requirement': '6', 'Time': '4:30pm - 5:00pm'}}, {'data_row_number': 30, 'values': {'Requirement': '7', 'Time': '5:00pm - 5:30pm'}}, {'data_row_number': 31, 'values': {'Requirement': '8', 'Time': '5:30pm - 6:00pm'}}, {'data_row_number': 32, 'values': {'Requirement': '9', 'Time': '6:00pm - 6:30pm'}}, {'data_row_number': 33, 'values': {'Requirement': '10', 'Time': '6:30pm - 7:00pm'}}, {'data_row_number': 34, 'values': {'Requirement': '9', 'Time': '7:00pm - 7:30pm'}}, {'data_row_number': 35, 'values': {'Requirement': '8', 'Time': '7:30pm - 8:00pm'}}, {'data_row_number': 36, 'values': {'Requirement': '7', 'Time': '8:00pm - 8:30pm'}}, {'data_row_number': 37, 'values': {'Requirement': '6', 'Time': '8:30pm - 9:00pm'}}, {'data_row_number': 38, 'values': {'Requirement': '5', 'Time': '9:00pm - 9:30pm'}}, {'data_row_number': 39, 'values': {'Requirement': '4', 'Time': '9:30pm - 10:00pm'}}, {'data_row_number': 40, 'values': {'Requirement': '4', 'Time': '10:00pm - 10:30pm'}}, {'data_row_number': 41, 'values': {'Requirement': '3', 'Time': '10:30pm - 11:00pm'}}, {'data_row_number': 42, 'values': {'Requirement': '3', 'Time': '11:00pm - 11:30pm'}}, {'data_row_number': 43, 'values': {'Requirement': '3', 'Time': '11:30pm - 12:00am'}}, {'data_row_number': 44, 'values': {'Requirement': '3', 'Time': '12:00am - 12:30am'}}, {'data_row_number': 45, 'values': {'Requirement': '4', 'Time': '12:30am - 1:00am'}}, {'data_row_number': 46, 'values': {'Requirement': '4', 'Time': '1:00am - 1:30am'}}, {'data_row_number': 47, 'values': {'Requirement': '4', 'Time': '1:30am - 2:00am'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    S = list(range(1, 49))
    R = {}
    for idx, rec in enumerate(records, start=1):
        val = rec['values']['Requirement']
        R[idx] = int(val)
    C = {}
    for s in S:
        C[s] = {}
        for t in S:
            diff = (t - s) % 48
            C[s][t] = 1 if 0 <= diff <= 15 else 0
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(S, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((x[s] for s in S)), GRB.MINIMIZE)
    for t in S:
        model.addConstr(gurobipy.quicksum((C[s][t] * x[s] for s in S)) >= R[t])
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
