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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 1,
             'file_name': 'project_parameters.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'ProjectDeadline', 'Value': '23'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
import re

def solve_problem():
    tables = CSVQA_DATA['tables']
    activities_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    param_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    activities = []
    ND = {}
    CD = {}
    CC = {}
    predecessors = {}
    for rec in activities_table['records']:
        vals = rec['values']
        act = vals['Activity']
        activities.append(act)
        ND[act] = int(vals['NormalDuration'])
        CD[act] = int(vals['CrashDuration'])
        CC[act] = int(vals['CrashCostPerDay'])
        pred_str = vals['Predecessors']
        if pred_str.strip() == '':
            predecessors[act] = set()
        else:
            preds = set((p.strip() for p in re.split('[,;]', pred_str) if p.strip()))
            predecessors[act] = preds
    deadline = None
    for rec in param_table['records']:
        if rec['values']['Parameter'] == 'ProjectDeadline':
            deadline = int(rec['values']['Value'])
            break
    if deadline is None:
        raise ValueError('ProjectDeadline not found in parameters')
    m = gurobipy.Model('ProjectCrashing')
    m.setParam('MIPGap', 1e-06)
    s = m.addVars(activities, lb=0, vtype=GRB.CONTINUOUS, name='')
    z = m.addVars(activities, lb=0, vtype=GRB.INTEGER, name='')
    T = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='T')
    m.setObjective(gurobipy.quicksum((CC[i] * z[i] for i in activities)), GRB.MINIMIZE)
    for i in activities:
        max_crash = ND[i] - CD[i]
        if max_crash < 0:
            raise ValueError(f'Negative crash limit for activity {i}')
        m.addConstr(z[i] <= max_crash, name=f'CrashBound_{i}')
    for i in activities:
        for p in predecessors[i]:
            if p not in activities:
                raise ValueError(f'Predecessor {p} of activity {i} not in activity set')
            m.addConstr(s[i] >= s[p] + ND[p] - z[p], name=f'Precedence_{p}_to_{i}')
    for i in activities:
        m.addConstr(T >= s[i] + ND[i] - z[i], name=f'Completion_{i}')
    m.addConstr(T <= deadline, name='Deadline')
    m.optimize()
    return m
m = solve_problem()