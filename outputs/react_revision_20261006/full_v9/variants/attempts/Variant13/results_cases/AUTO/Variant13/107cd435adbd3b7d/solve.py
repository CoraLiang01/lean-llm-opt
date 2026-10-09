CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'An installation project consists of activities with precedence relationships. Normal duration, crash '
          'duration, and crash cost per day are listed in project_activities.csv; the required project deadline is '
          'listed in project_parameters.csv. Activity durations can be shortened only by integer numbers of days '
          'within the listed crash limits.\n'
          '\n'
          'Formulate a minimum-cost project crashing model. For each activity i, define s_i as the nonnegative start '
          'time and z_i as the integer number of crash days used; define T as the project completion time. The '
          'objective is to minimize total crashing cost. The model should include precedence constraints using crashed '
          'durations, crash-day lower and upper bounds, project-completion constraints for all activities, the '
          'project-deadline constraint, nonnegativity constraints for start and completion-time variables, and integer '
          'restrictions for crash variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Activity', 'Predecessors', 'NormalDuration', 'CrashDuration', 'CrashCostPerDay'],
             'file_index': 0,
             'file_name': 'project_activities.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'Activity': 'A',
                                     'CrashCostPerDay': '180',
                                     'CrashDuration': '3',
                                     'NormalDuration': '5',
                                     'Predecessors': ''}},
                         {'source_row': 1,
                          'values': {'Activity': 'B',
                                     'CrashCostPerDay': '150',
                                     'CrashDuration': '4',
                                     'NormalDuration': '6',
                                     'Predecessors': ''}},
                         {'source_row': 2,
                          'values': {'Activity': 'C',
                                     'CrashCostPerDay': '130',
                                     'CrashDuration': '4',
                                     'NormalDuration': '7',
                                     'Predecessors': 'A'}},
                         {'source_row': 3,
                          'values': {'Activity': 'D',
                                     'CrashCostPerDay': '210',
                                     'CrashDuration': '3',
                                     'NormalDuration': '4',
                                     'Predecessors': 'A'}},
                         {'source_row': 4,
                          'values': {'Activity': 'E',
                                     'CrashCostPerDay': '160',
                                     'CrashDuration': '3',
                                     'NormalDuration': '5',
                                     'Predecessors': 'B'}},
                         {'source_row': 5,
                          'values': {'Activity': 'F',
                                     'CrashCostPerDay': '190',
                                     'CrashDuration': '4',
                                     'NormalDuration': '6',
                                     'Predecessors': 'C;D'}},
                         {'source_row': 6,
                          'values': {'Activity': 'G',
                                     'CrashCostPerDay': '140',
                                     'CrashDuration': '5',
                                     'NormalDuration': '7',
                                     'Predecessors': 'D;E'}},
                         {'source_row': 7,
                          'values': {'Activity': 'H',
                                     'CrashCostPerDay': '220',
                                     'CrashDuration': '2',
                                     'NormalDuration': '4',
                                     'Predecessors': 'F'}},
                         {'source_row': 8,
                          'values': {'Activity': 'I',
                                     'CrashCostPerDay': '170',
                                     'CrashDuration': '3',
                                     'NormalDuration': '5',
                                     'Predecessors': 'G'}},
                         {'source_row': 9,
                          'values': {'Activity': 'J',
                                     'CrashCostPerDay': '260',
                                     'CrashDuration': '2',
                                     'NormalDuration': '3',
                                     'Predecessors': 'H;I'}}],
             'returned_rows': 10,
             'role': 'project activities',
             'table_id': 'file_0_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 1,
             'file_name': 'project_parameters.csv',
             'filters': {'conditions': [{'column': 'Parameter',
                                         'dtype': 'string',
                                         'evidence': 'the required project deadline is listed in '
                                                     'project_parameters.csv',
                                         'operator': 'exact',
                                         'value': 'ProjectDeadline'}],
                         'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'ProjectDeadline', 'Value': '23'}}],
             'returned_rows': 1,
             'role': 'project parameters',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    activities_df = CSVQA_FRAMES['file_0_view_0']
    params_df = CSVQA_FRAMES['file_1_view_0']
    I = []
    d_norm = {}
    d_crash = {}
    c = {}
    P = {}
    for (idx, row) in activities_df.iterrows():
        act = row['Activity']
        I.append(act)
        d_norm[act] = float(row['NormalDuration'])
        d_crash[act] = float(row['CrashDuration'])
        c[act] = float(row['CrashCostPerDay'])
        preds = row['Predecessors']
        if preds.strip() == '':
            P[act] = []
        else:
            P[act] = [p.strip() for p in preds.split(';') if p.strip() != '']
    D = None
    for (idx, row) in params_df.iterrows():
        if row['Parameter'].casefold() == 'projectdeadline':
            D = float(row['Value'])
            break
    if D is None:
        raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
    m = gp.Model('ProjectCrashing')
    s_vars = m.addVars(I, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z_vars = m.addVars(I, lb=0, ub=None, vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    for i in I:
        ub = d_norm[i] - d_crash[i]
        m.addConstr(z_vars[i] <= ub, name=f'crash_ub_{i}')
        m.addConstr(z_vars[i] >= 0, name=f'crash_lb_{i}')
    for i in I:
        for j in P[i]:
            m.addConstr(s_vars[i] >= s_vars[j] + d_norm[j] - z_vars[j], name=f'prec_{i}_{j}')
    for i in I:
        m.addConstr(T >= s_vars[i] + d_norm[i] - z_vars[i], name=f'completion_{i}')
    m.addConstr(T <= D, name='deadline')
    m.setObjective(gp.quicksum((c[i] * z_vars[i] for i in I)), gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)