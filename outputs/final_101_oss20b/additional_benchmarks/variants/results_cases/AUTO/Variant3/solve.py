CSVQA_DATA = {'query': "A construction firm is planning a project made up of multiple activities with precedence relationships. The activity data are provided in project_activities.csv, including each activity's immediate predecessors, normal duration, shortest crash duration, and crash cost per day. The project deadline is provided in project_parameters.csv. The firm may reduce an activity's duration by paying the corresponding crash cost per day, but no activity can be shortened below its crash duration.\n\nFormulate a mixed-integer project-crashing model. For each activity i, define s_i as the start time and z_i as the integer number of days by which activity i is crashed. Define T as the project completion time. The objective is to minimize total crashing cost. The model should include precedence constraints using the crashed activity durations, bounds on each crashing variable, project-completion constraints for all activities, the project deadline constraint, nonnegativity constraints for start times, and integer restrictions for the crashing variables.", 'relationships': [], 'tables': [{'columns': ['Activity', 'Predecessors', 'NormalDuration', 'CrashDuration', 'CrashCostPerDay'], 'file_name': 'project_activities.csv', 'records': [{'data_row_number': 0, 'values': {'Activity': 'A', 'CrashCostPerDay': '300', 'CrashDuration': '4', 'NormalDuration': '6', 'Predecessors': ''}}, {'data_row_number': 1, 'values': {'Activity': 'B', 'CrashCostPerDay': '250', 'CrashDuration': '4', 'NormalDuration': '5', 'Predecessors': ''}}, {'data_row_number': 2, 'values': {'Activity': 'C', 'CrashCostPerDay': '180', 'CrashDuration': '4', 'NormalDuration': '7', 'Predecessors': 'A'}}, {'data_row_number': 3, 'values': {'Activity': 'D', 'CrashCostPerDay': '220', 'CrashDuration': '3', 'NormalDuration': '4', 'Predecessors': 'A'}}, {'data_row_number': 4, 'values': {'Activity': 'E', 'CrashCostPerDay': '160', 'CrashDuration': '4', 'NormalDuration': '6', 'Predecessors': 'B'}}, {'data_row_number': 5, 'values': {'Activity': 'F', 'CrashCostPerDay': '140', 'CrashDuration': '5', 'NormalDuration': '8', 'Predecessors': 'B'}}, {'data_row_number': 6, 'values': {'Activity': 'G', 'CrashCostPerDay': '210', 'CrashDuration': '3', 'NormalDuration': '5', 'Predecessors': 'C;D'}}, {'data_row_number': 7, 'values': {'Activity': 'H', 'CrashCostPerDay': '190', 'CrashDuration': '4', 'NormalDuration': '7', 'Predecessors': 'D;E'}}, {'data_row_number': 8, 'values': {'Activity': 'I', 'CrashCostPerDay': '170', 'CrashDuration': '4', 'NormalDuration': '6', 'Predecessors': 'F'}}, {'data_row_number': 9, 'values': {'Activity': 'J', 'CrashCostPerDay': '260', 'CrashDuration': '3', 'NormalDuration': '4', 'Predecessors': 'G;H'}}, {'data_row_number': 10, 'values': {'Activity': 'K', 'CrashCostPerDay': '150', 'CrashDuration': '3', 'NormalDuration': '5', 'Predecessors': 'H;I'}}, {'data_row_number': 11, 'values': {'Activity': 'L', 'CrashCostPerDay': '320', 'CrashDuration': '2', 'NormalDuration': '3', 'Predecessors': 'J;K'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Parameter', 'Value'], 'file_name': 'project_parameters.csv', 'records': [{'data_row_number': 0, 'values': {'Parameter': 'ProjectDeadline', 'Value': '23'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    act_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    param_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    records = act_table['records']
    activities = [rec['values']['Activity'] for rec in records]
    ND = {}
    CD = {}
    CC = {}
    P = {}
    for rec in records:
        act = rec['values']['Activity']
        try:
            ND[act] = float(rec['values']['NormalDuration'])
            CD[act] = float(rec['values']['CrashDuration'])
            CC[act] = float(rec['values']['CrashCostPerDay'])
        except KeyError as e:
            raise ValueError(f'Missing parameter in activity record {act}: {e}')
        preds = rec['values']['Predecessors']
        if preds:
            pred_list = [p.strip() for p in preds.split(';') if p.strip()]
        else:
            pred_list = []
        P[act] = pred_list
    param_records = param_table['records']
    D = None
    for rec in param_records:
        if rec['values']['Parameter'] == 'ProjectDeadline':
            D = float(rec['values']['Value'])
            break
    if D is None:
        raise ValueError('ProjectDeadline not found in parameters')
    for act in activities:
        if ND[act] < CD[act]:
            raise ValueError(f'Normal duration less than crash duration for activity {act}')
        for p in P[act]:
            if p not in activities:
                raise ValueError(f'Predecessor {p} of activity {act} not defined')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    s = model.addVars(activities, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='')
    z = model.addVars(activities, lb=0, vtype=gurobipy.GRB.INTEGER, name='')
    T = model.addVar(lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='T')
    model.setObjective(gurobipy.quicksum((CC[i] * z[i] for i in activities)), gurobipy.GRB.MINIMIZE)
    for i in activities:
        model.addConstr(z[i] <= ND[i] - CD[i], name=f'CrashBound_{i}')
    for i in activities:
        for p in P[i]:
            model.addConstr(s[i] >= s[p] + (ND[p] - z[p]), name=f'Precedence_{p}_{i}')
    for i in activities:
        model.addConstr(T >= s[i] + (ND[i] - z[i]), name=f'Completion_{i}')
    model.addConstr(T <= D, name='Deadline')
    model.optimize()
    if model.Status == gurobipy.GRB.OPTIMAL:
        print(f'Optimal objective: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName} = {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
m
