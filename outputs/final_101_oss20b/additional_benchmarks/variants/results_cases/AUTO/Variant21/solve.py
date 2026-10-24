CSVQA_DATA = {'query': 'A renovation project is made up of precedence-linked activities. Normal duration, crash duration, and crash cost per day are listed in project_activities.csv; the required completion deadline is listed in project_parameters.csv. Activity durations may be shortened by integer numbers of days within the listed crash limits.\n\nFormulate a minimum-cost project crashing model. For each activity i, define s_i as the nonnegative start time and z_i as the integer number of crash days used; define T as the project completion time. The objective is to minimize total crashing cost. The model should include precedence constraints using crashed durations, crash-day lower and upper bounds, project-completion constraints for all activities, the project-deadline constraint, nonnegativity constraints for start and completion-time variables, and integer restrictions for crash variables.', 'relationships': [], 'tables': [{'columns': ['Activity', 'Predecessors', 'NormalDuration', 'CrashDuration', 'CrashCostPerDay'], 'file_name': 'project_activities.csv', 'records': [{'data_row_number': 0, 'values': {'Activity': 'A', 'CrashCostPerDay': '120', 'CrashDuration': '3', 'NormalDuration': '4', 'Predecessors': ''}}, {'data_row_number': 1, 'values': {'Activity': 'B', 'CrashCostPerDay': '150', 'CrashDuration': '5', 'NormalDuration': '7', 'Predecessors': ''}}, {'data_row_number': 2, 'values': {'Activity': 'C', 'CrashCostPerDay': '180', 'CrashDuration': '4', 'NormalDuration': '6', 'Predecessors': 'A'}}, {'data_row_number': 3, 'values': {'Activity': 'D', 'CrashCostPerDay': '160', 'CrashDuration': '3', 'NormalDuration': '5', 'Predecessors': 'A'}}, {'data_row_number': 4, 'values': {'Activity': 'E', 'CrashCostPerDay': '210', 'CrashDuration': '2', 'NormalDuration': '4', 'Predecessors': 'B'}}, {'data_row_number': 5, 'values': {'Activity': 'F', 'CrashCostPerDay': '140', 'CrashDuration': '5', 'NormalDuration': '7', 'Predecessors': 'C;D'}}, {'data_row_number': 6, 'values': {'Activity': 'G', 'CrashCostPerDay': '170', 'CrashDuration': '4', 'NormalDuration': '6', 'Predecessors': 'D;E'}}, {'data_row_number': 7, 'values': {'Activity': 'H', 'CrashCostPerDay': '200', 'CrashDuration': '3', 'NormalDuration': '5', 'Predecessors': 'F;G'}}, {'data_row_number': 8, 'values': {'Activity': 'I', 'CrashCostPerDay': '260', 'CrashDuration': '2', 'NormalDuration': '3', 'Predecessors': 'H'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Parameter', 'Value'], 'file_name': 'project_parameters.csv', 'records': [{'data_row_number': 0, 'values': {'Parameter': 'ProjectDeadline', 'Value': '22'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    activities_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
    params_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
    records = activities_table['records']
    I = [rec['values']['Activity'] for rec in records]
    ND = {}
    CD = {}
    CC = {}
    P = {}
    for rec in records:
        act = rec['values']['Activity']
        try:
            ND[act] = int(rec['values']['NormalDuration'])
            CD[act] = int(rec['values']['CrashDuration'])
            CC[act] = int(rec['values']['CrashCostPerDay'])
        except KeyError as e:
            raise ValueError(f'Missing parameter in activity {act}: {e}')
        preds_str = rec['values']['Predecessors']
        if preds_str.strip() == '':
            P[act] = []
        else:
            preds = [p.strip() for p in preds_str.split(';') if p.strip()]
            for p in preds:
                if p not in I:
                    raise ValueError(f'Predecessor {p} of activity {act} not in activity set')
            P[act] = preds
    U = {}
    for i in I:
        u = ND[i] - CD[i]
        if u < 0:
            raise ValueError(f'Negative crash days for activity {i}')
        U[i] = u
    param_records = params_table['records']
    if not param_records:
        raise ValueError('Project deadline parameter missing')
    D = int(param_records[0]['values']['Value'])
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    s = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    z = m.addVars(I, lb=0, ub=U, vtype=GRB.INTEGER, name='')
    T = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((CC[i] * z[i] for i in I)), GRB.MINIMIZE)
    for j in I:
        for i in P[j]:
            m.addConstr(s[j] >= s[i] + ND[i] - z[i], name=f'prec_{i}_{j}')
    for i in I:
        m.addConstr(T >= s[i] + ND[i] - z[i], name=f'comp_{i}')
    m.addConstr(T <= D, name='deadline')
    m.optimize()
    return m
m = solve_problem()
