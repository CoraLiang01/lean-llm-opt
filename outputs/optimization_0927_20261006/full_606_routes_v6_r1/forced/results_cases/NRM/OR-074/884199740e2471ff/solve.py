CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A certain restaurant operates 24 hours a day and requires a minimum number of waitstaff as shown in '
          '44.csv.\n'
          '\n'
          '    Each waitstaff works continuously for 8 hours a day. The goal is to find the minimum number of '
          'waitstaff that satisfies the above conditions and represent this problem as a linear programming model.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Time', 'Requirement'],
             'file_index': 0,
             'file_name': '44.csv',
             'filters': {},
             'original_rows': 48,
             'records': [{'source_row': 0, 'values': {'Requirement': '2', 'Time': '2:00am - 2:30am'}},
                         {'source_row': 1, 'values': {'Requirement': '3', 'Time': '2:30am - 3:00am'}},
                         {'source_row': 2, 'values': {'Requirement': '4', 'Time': '3:00am - 3:30am'}},
                         {'source_row': 3, 'values': {'Requirement': '6', 'Time': '3:30am - 4:00am'}},
                         {'source_row': 4, 'values': {'Requirement': '5', 'Time': '4:00am - 4:30am'}},
                         {'source_row': 5, 'values': {'Requirement': '4', 'Time': '4:30am - 5:00am'}},
                         {'source_row': 6, 'values': {'Requirement': '5', 'Time': '5:00am - 5:30am'}},
                         {'source_row': 7, 'values': {'Requirement': '6', 'Time': '5:30am - 6:00am'}},
                         {'source_row': 8, 'values': {'Requirement': '7', 'Time': '6:00am - 6:30am'}},
                         {'source_row': 9, 'values': {'Requirement': '8', 'Time': '6:30am - 7:00am'}},
                         {'source_row': 10, 'values': {'Requirement': '9', 'Time': '7:00am - 7:30am'}},
                         {'source_row': 11, 'values': {'Requirement': '9', 'Time': '7:30am - 8:00am'}},
                         {'source_row': 12, 'values': {'Requirement': '8', 'Time': '8:00am - 8:30am'}},
                         {'source_row': 13, 'values': {'Requirement': '8', 'Time': '8:30am - 9:00am'}},
                         {'source_row': 14, 'values': {'Requirement': '9', 'Time': '9:00am - 9:30am'}},
                         {'source_row': 15, 'values': {'Requirement': '9', 'Time': '9:30am - 10:00am'}},
                         {'source_row': 16, 'values': {'Requirement': '10', 'Time': '10:00am - 10:30am'}},
                         {'source_row': 17, 'values': {'Requirement': '12', 'Time': '10:30am - 11:00am'}},
                         {'source_row': 18, 'values': {'Requirement': '11', 'Time': '11:00am - 11:30am'}},
                         {'source_row': 19, 'values': {'Requirement': '11', 'Time': '11:30am - 12:00pm'}},
                         {'source_row': 20, 'values': {'Requirement': '12', 'Time': '12:00pm - 12:30pm'}},
                         {'source_row': 21, 'values': {'Requirement': '11', 'Time': '12:30pm - 1:00pm'}},
                         {'source_row': 22, 'values': {'Requirement': '10', 'Time': '1:00pm - 1:30pm'}},
                         {'source_row': 23, 'values': {'Requirement': '9', 'Time': '1:30pm - 2:00pm'}},
                         {'source_row': 24, 'values': {'Requirement': '8', 'Time': '2:00pm - 2:30pm'}},
                         {'source_row': 25, 'values': {'Requirement': '7', 'Time': '2:30pm - 3:00pm'}},
                         {'source_row': 26, 'values': {'Requirement': '6', 'Time': '3:00pm - 3:30pm'}},
                         {'source_row': 27, 'values': {'Requirement': '5', 'Time': '3:30pm - 4:00pm'}},
                         {'source_row': 28, 'values': {'Requirement': '5', 'Time': '4:00pm - 4:30pm'}},
                         {'source_row': 29, 'values': {'Requirement': '6', 'Time': '4:30pm - 5:00pm'}},
                         {'source_row': 30, 'values': {'Requirement': '7', 'Time': '5:00pm - 5:30pm'}},
                         {'source_row': 31, 'values': {'Requirement': '8', 'Time': '5:30pm - 6:00pm'}},
                         {'source_row': 32, 'values': {'Requirement': '9', 'Time': '6:00pm - 6:30pm'}},
                         {'source_row': 33, 'values': {'Requirement': '10', 'Time': '6:30pm - 7:00pm'}},
                         {'source_row': 34, 'values': {'Requirement': '9', 'Time': '7:00pm - 7:30pm'}},
                         {'source_row': 35, 'values': {'Requirement': '8', 'Time': '7:30pm - 8:00pm'}},
                         {'source_row': 36, 'values': {'Requirement': '7', 'Time': '8:00pm - 8:30pm'}},
                         {'source_row': 37, 'values': {'Requirement': '6', 'Time': '8:30pm - 9:00pm'}},
                         {'source_row': 38, 'values': {'Requirement': '5', 'Time': '9:00pm - 9:30pm'}},
                         {'source_row': 39, 'values': {'Requirement': '4', 'Time': '9:30pm - 10:00pm'}},
                         {'source_row': 40, 'values': {'Requirement': '4', 'Time': '10:00pm - 10:30pm'}},
                         {'source_row': 41, 'values': {'Requirement': '3', 'Time': '10:30pm - 11:00pm'}},
                         {'source_row': 42, 'values': {'Requirement': '3', 'Time': '11:00pm - 11:30pm'}},
                         {'source_row': 43, 'values': {'Requirement': '3', 'Time': '11:30pm - 12:00am'}},
                         {'source_row': 44, 'values': {'Requirement': '3', 'Time': '12:00am - 12:30am'}},
                         {'source_row': 45, 'values': {'Requirement': '4', 'Time': '12:30am - 1:00am'}},
                         {'source_row': 46, 'values': {'Requirement': '4', 'Time': '1:00am - 1:30am'}},
                         {'source_row': 47, 'values': {'Requirement': '4', 'Time': '1:30am - 2:00am'}}],
             'returned_rows': 48,
             'role': 'waitstaff minimum requirements by time interval',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Table with table_id 'file_0_view_0' not found.")
records = table['records']
if len(records) != 48:
    raise ValueError('Expected 48 records for 48 half-hour intervals.')
T = [i for i in range(48)]
S = [i for i in range(48)]
time_labels = {}
requirements = {}
for (i, rec) in enumerate(records):
    time_labels[i] = rec['values']['Time']
    try:
        requirements[i] = int(rec['values']['Requirement'])
    except Exception:
        raise ValueError(f'Requirement at row {i} is not an integer.')
n = 16
if set(requirements.keys()) != set(T):
    raise ValueError('Missing requirement data for some intervals.')
shift_covers = {}
for s in S:
    shift_covers[s] = [(s + k) % 48 for k in range(n)]
interval_covered_by = {t: [] for t in T}
for s in S:
    for t_cov in shift_covers[s]:
        interval_covered_by[t_cov].append(s)
m = gp.Model('Waitstaff_Shift_Scheduling')
x_vars = m.addVars(S, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in S)), GRB.MINIMIZE)
for t in T:
    covering_shifts = interval_covered_by[t]
    if not covering_shifts:
        raise ValueError(f'No shifts cover interval {t}.')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirements[t], name=f'req_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for s in S:
        print(f'x[{s}]: {x_vars[s].X}')
else:
    print(f'Solver status: {m.Status}')