CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A renovation project is made up of precedence-linked activities. Normal duration, crash duration, and crash '
          'cost per day are listed in project_activities.csv; the required completion deadline is listed in '
          'project_parameters.csv. Activity durations may be shortened by integer numbers of days within the listed '
          'crash limits.\n'
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
             'original_rows': 9,
             'records': [{'source_row': 0,
                          'values': {'Activity': 'A',
                                     'CrashCostPerDay': '120',
                                     'CrashDuration': '3',
                                     'NormalDuration': '4',
                                     'Predecessors': ''}},
                         {'source_row': 1,
                          'values': {'Activity': 'B',
                                     'CrashCostPerDay': '150',
                                     'CrashDuration': '5',
                                     'NormalDuration': '7',
                                     'Predecessors': ''}},
                         {'source_row': 2,
                          'values': {'Activity': 'C',
                                     'CrashCostPerDay': '180',
                                     'CrashDuration': '4',
                                     'NormalDuration': '6',
                                     'Predecessors': 'A'}},
                         {'source_row': 3,
                          'values': {'Activity': 'D',
                                     'CrashCostPerDay': '160',
                                     'CrashDuration': '3',
                                     'NormalDuration': '5',
                                     'Predecessors': 'A'}},
                         {'source_row': 4,
                          'values': {'Activity': 'E',
                                     'CrashCostPerDay': '210',
                                     'CrashDuration': '2',
                                     'NormalDuration': '4',
                                     'Predecessors': 'B'}},
                         {'source_row': 5,
                          'values': {'Activity': 'F',
                                     'CrashCostPerDay': '140',
                                     'CrashDuration': '5',
                                     'NormalDuration': '7',
                                     'Predecessors': 'C;D'}},
                         {'source_row': 6,
                          'values': {'Activity': 'G',
                                     'CrashCostPerDay': '170',
                                     'CrashDuration': '4',
                                     'NormalDuration': '6',
                                     'Predecessors': 'D;E'}},
                         {'source_row': 7,
                          'values': {'Activity': 'H',
                                     'CrashCostPerDay': '200',
                                     'CrashDuration': '3',
                                     'NormalDuration': '5',
                                     'Predecessors': 'F;G'}},
                         {'source_row': 8,
                          'values': {'Activity': 'I',
                                     'CrashCostPerDay': '260',
                                     'CrashDuration': '2',
                                     'NormalDuration': '3',
                                     'Predecessors': 'H'}}],
             'returned_rows': 9,
             'role': 'activity data',
             'table_id': 'file_0_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 1,
             'file_name': 'project_parameters.csv',
             'filters': {'conditions': [{'column': 'Parameter',
                                         'dtype': 'string',
                                         'evidence': 'the required completion deadline is listed in '
                                                     'project_parameters.csv',
                                         'operator': 'exact',
                                         'value': 'ProjectDeadline'}],
                         'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'ProjectDeadline', 'Value': '22'}}],
             'returned_rows': 1,
             'role': 'project parameters',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    activities_frame = CSVQA_FRAMES['file_0_view_0']
    parameters_frame = CSVQA_FRAMES['file_1_view_0']
    activities = []
    d_norm = {}
    d_crash = {}
    c = {}
    preds = {}
    for (idx, row) in activities_frame.iterrows():
        act = row['Activity']
        activities.append(act)
        try:
            d_norm[act] = float(row['NormalDuration'])
            d_crash[act] = float(row['CrashDuration'])
            c[act] = float(row['CrashCostPerDay'])
        except Exception as e:
            raise ValueError(f'Non-numeric duration/cost for activity {act}: {e}')
        pred_field = row['Predecessors']
        if pred_field.strip() == '':
            preds[act] = []
        else:
            preds[act] = [p.strip() for p in pred_field.split(';') if p.strip() != '']
    D = None
    for (idx, row) in parameters_frame.iterrows():
        if row['Parameter'].casefold() == 'projectdeadline':
            try:
                D = float(row['Value'])
            except Exception as e:
                raise ValueError(f'Non-numeric ProjectDeadline: {e}')
    if D is None:
        raise ValueError('ProjectDeadline not found in project_parameters.csv')
    for i in activities:
        if d_norm[i] < d_crash[i]:
            raise ValueError(f'NormalDuration < CrashDuration for activity {i}')
    m = gp.Model('ProjectCrashing')
    s_vars = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z_vars = m.addVars(activities, lb=0, ub={i: d_norm[i] - d_crash[i] for i in activities}, vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((c[i] * z_vars[i] for i in activities)), gp.GRB.MINIMIZE)
    for i in activities:
        for j in preds[i]:
            m.addConstr(s_vars[i] >= s_vars[j] + d_norm[j] - z_vars[j], name=f'prec_{i}_{j}')
    for i in activities:
        m.addConstr(z_vars[i] >= 0, name=f'z_lb_{i}')
        m.addConstr(z_vars[i] <= d_norm[i] - d_crash[i], name=f'z_ub_{i}')
    for i in activities:
        m.addConstr(T >= s_vars[i] + d_norm[i] - z_vars[i], name=f'T_comp_{i}')
    m.addConstr(T <= D, name='deadline')
    for i in activities:
        m.addConstr(s_vars[i] >= 0, name=f's_lb_{i}')
    m.addConstr(T >= 0, name='T_lb')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)