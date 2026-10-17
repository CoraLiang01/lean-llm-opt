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
             'filters': {},
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
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'ProjectDeadline', 'Value': '23'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
data = CSVQA_DATA
activities_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
activities_records = activities_table['records']
parameters_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
parameters_records = parameters_table['records']
activities = []
normal_dur = {}
crash_dur = {}
crash_cost = {}
predecessors = {}
for rec in activities_records:
    vals = rec['values']
    act = vals['Activity']
    activities.append(act)
    normal_dur[act] = float(vals['NormalDuration'])
    crash_dur[act] = float(vals['CrashDuration'])
    crash_cost[act] = float(vals['CrashCostPerDay'])
    preds_str = vals['Predecessors']
    if preds_str.strip() == '':
        predecessors[act] = []
    else:
        predecessors[act] = [p.strip() for p in preds_str.split(';') if p.strip()]
max_crash_dict = {act: int(normal_dur[act] - crash_dur[act]) for act in activities}
deadline = None
for rec in parameters_records:
    if rec['values']['Parameter'] == 'ProjectDeadline':
        deadline = float(rec['values']['Value'])
        break
if deadline is None:
    raise ValueError('ProjectDeadline parameter not found')
for act, preds in predecessors.items():
    for p in preds:
        if p not in activities:
            raise ValueError(f'Predecessor {p} of activity {act} not found')
m = gurobipy.Model('ProjectCrashing')
m.setParam('MIPGap', 1e-06)
s = m.addVars(activities, lb=0, vtype=GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, ub=max_crash_dict, vtype=GRB.INTEGER, name='')
T = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='T')
m.setObjective(gurobipy.quicksum((crash_cost[i] * z[i] for i in activities)), GRB.MINIMIZE)
for i in activities:
    for p in predecessors[i]:
        m.addConstr(s[i] >= s[p] + normal_dur[p] - z[p], name=f'prec_{i}_{p}')
    m.addConstr(T >= s[i] + normal_dur[i] - z[i], name=f'comp_{i}')
m.addConstr(T <= deadline, name='deadline')
m.optimize()