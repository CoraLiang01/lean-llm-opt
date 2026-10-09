CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A construction firm is planning a project made up of multiple activities with precedence relationships. The '
          "activity data are provided in project_activities.csv, including each activity's immediate predecessors, "
          'normal duration, shortest crash duration, and crash cost per day. The project deadline is provided in '
          "project_parameters.csv. The firm may reduce an activity's duration by paying the corresponding crash cost "
          'per day, but no activity can be shortened below its crash duration.\n'
          '\n'
          'Formulate a mixed-integer project-crashing model. For each activity i, define s_i as the start time and z_i '
          'as the integer number of days by which activity i is crashed. Define T as the project completion time. The '
          'objective is to minimize total crashing cost. The model should include precedence constraints using the '
          'crashed activity durations, bounds on each crashing variable, project-completion constraints for all '
          'activities, the project deadline constraint, nonnegativity constraints for start times, and integer '
          'restrictions for the crashing variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Activity', 'Predecessors', 'NormalDuration', 'CrashDuration', 'CrashCostPerDay'],
             'file_index': 0,
             'file_name': 'project_activities.csv',
             'filters': {},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Activity': 'A',
                                     'CrashCostPerDay': '300',
                                     'CrashDuration': '4',
                                     'NormalDuration': '6',
                                     'Predecessors': ''}},
                         {'source_row': 1,
                          'values': {'Activity': 'B',
                                     'CrashCostPerDay': '250',
                                     'CrashDuration': '4',
                                     'NormalDuration': '5',
                                     'Predecessors': ''}},
                         {'source_row': 2,
                          'values': {'Activity': 'C',
                                     'CrashCostPerDay': '180',
                                     'CrashDuration': '4',
                                     'NormalDuration': '7',
                                     'Predecessors': 'A'}},
                         {'source_row': 3,
                          'values': {'Activity': 'D',
                                     'CrashCostPerDay': '220',
                                     'CrashDuration': '3',
                                     'NormalDuration': '4',
                                     'Predecessors': 'A'}},
                         {'source_row': 4,
                          'values': {'Activity': 'E',
                                     'CrashCostPerDay': '160',
                                     'CrashDuration': '4',
                                     'NormalDuration': '6',
                                     'Predecessors': 'B'}},
                         {'source_row': 5,
                          'values': {'Activity': 'F',
                                     'CrashCostPerDay': '140',
                                     'CrashDuration': '5',
                                     'NormalDuration': '8',
                                     'Predecessors': 'B'}},
                         {'source_row': 6,
                          'values': {'Activity': 'G',
                                     'CrashCostPerDay': '210',
                                     'CrashDuration': '3',
                                     'NormalDuration': '5',
                                     'Predecessors': 'C;D'}},
                         {'source_row': 7,
                          'values': {'Activity': 'H',
                                     'CrashCostPerDay': '190',
                                     'CrashDuration': '4',
                                     'NormalDuration': '7',
                                     'Predecessors': 'D;E'}},
                         {'source_row': 8,
                          'values': {'Activity': 'I',
                                     'CrashCostPerDay': '170',
                                     'CrashDuration': '4',
                                     'NormalDuration': '6',
                                     'Predecessors': 'F'}},
                         {'source_row': 9,
                          'values': {'Activity': 'J',
                                     'CrashCostPerDay': '260',
                                     'CrashDuration': '3',
                                     'NormalDuration': '4',
                                     'Predecessors': 'G;H'}},
                         {'source_row': 10,
                          'values': {'Activity': 'K',
                                     'CrashCostPerDay': '150',
                                     'CrashDuration': '3',
                                     'NormalDuration': '5',
                                     'Predecessors': 'H;I'}},
                         {'source_row': 11,
                          'values': {'Activity': 'L',
                                     'CrashCostPerDay': '320',
                                     'CrashDuration': '2',
                                     'NormalDuration': '3',
                                     'Predecessors': 'J;K'}}],
             'returned_rows': 12,
             'role': 'activity data',
             'table_id': 'file_0_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 1,
             'file_name': 'project_parameters.csv',
             'filters': {'conditions': [{'column': 'Parameter',
                                         'dtype': 'string',
                                         'evidence': 'The project deadline is provided in project_parameters.csv.',
                                         'inclusive': 'both',
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
import numpy as np
import sys
import re

def solve_problem(CSVQA_FRAMES):
    activities_df = CSVQA_FRAMES['file_0_view_0']
    params_df = CSVQA_FRAMES['file_1_view_0']
    activities = []
    d_norm = {}
    d_crash = {}
    c = {}
    preds = {}
    for (idx, row) in activities_df.iterrows():
        act = row['Activity']
        activities.append(act)
        try:
            d_norm[act] = float(row['NormalDuration'])
            d_crash[act] = float(row['CrashDuration'])
            c[act] = float(row['CrashCostPerDay'])
        except Exception as e:
            raise ValueError(f'Non-numeric duration or cost for activity {act}: {e}')
        pred_field = row['Predecessors']
        if isinstance(pred_field, str) and pred_field.strip() != '':
            preds[act] = [p.strip() for p in pred_field.split(';') if p.strip() != '']
        else:
            preds[act] = []
    deadline_row = params_df[params_df['Parameter'].str.casefold() == 'projectdeadline']
    if deadline_row.empty:
        raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
    D = float(deadline_row.iloc[0]['Value'])
    for act in activities:
        if act not in d_norm or act not in d_crash or act not in c or (act not in preds):
            raise ValueError(f'Missing data for activity {act}')
    m = gp.Model('ProjectCrashing')
    s_vars = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z_vars = m.addVars(activities, lb=0.0, ub={i: d_norm[i] - d_crash[i] for i in activities}, vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((c[i] * z_vars[i] for i in activities)), gp.GRB.MINIMIZE)
    for i in activities:
        for j in preds[i]:
            if j not in activities:
                raise ValueError(f'Predecessor {j} of activity {i} not found in activity list')
            m.addConstr(s_vars[i] >= s_vars[j] + d_norm[j] - z_vars[j], name=f'prec_{i}_{j}')
    for i in activities:
        m.addConstr(z_vars[i] >= 0, name=f'z_lb_{i}')
        m.addConstr(z_vars[i] <= d_norm[i] - d_crash[i], name=f'z_ub_{i}')
    for i in activities:
        m.addConstr(s_vars[i] >= 0, name=f's_lb_{i}')
    for i in activities:
        m.addConstr(T >= s_vars[i] + d_norm[i] - z_vars[i], name=f'T_ge_{i}')
    m.addConstr(T <= D, name='deadline')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)