CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Now we need to determine 10 out of 12 workers to each complete one of the ten tasks. Since each worker has '
          'different skill sets, the amount of time required for each worker to complete each task is also different. '
          'The time required for each worker to complete each task is shown in 15.csv.\n'
          '\n'
          '    Try to find a work assignment plan that minimizes the total working hours.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Task Time Required', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
             'file_index': 0,
             'file_name': '15.csv',
             'filters': {},
             'original_rows': 13,
             'records': [{'source_row': 0,
                          'values': {'A': '',
                                     'B': '',
                                     'C': '',
                                     'D': '',
                                     'E': '',
                                     'F': '',
                                     'G': '',
                                     'H': '',
                                     'I': '',
                                     'J': '',
                                     'Task Time Required': 'Worker'}},
                         {'source_row': 1,
                          'values': {'A': '9',
                                     'B': '4',
                                     'C': '3',
                                     'D': '7',
                                     'E': '6',
                                     'F': '5',
                                     'G': '6',
                                     'H': '3',
                                     'I': '7',
                                     'J': '5',
                                     'Task Time Required': '1'}},
                         {'source_row': 2,
                          'values': {'A': '4',
                                     'B': '6',
                                     'C': '5',
                                     'D': '6',
                                     'E': '4',
                                     'F': '5',
                                     'G': '3',
                                     'H': '8',
                                     'I': '7',
                                     'J': '6',
                                     'Task Time Required': '2'}},
                         {'source_row': 3,
                          'values': {'A': '5',
                                     'B': '4',
                                     'C': '7',
                                     'D': '5',
                                     'E': '6',
                                     'F': '6',
                                     'G': '5',
                                     'H': '8',
                                     'I': '6',
                                     'J': '9',
                                     'Task Time Required': '3'}},
                         {'source_row': 4,
                          'values': {'A': '7',
                                     'B': '5',
                                     'C': '2',
                                     'D': '3',
                                     'E': '7',
                                     'F': '8',
                                     'G': '5',
                                     'H': '6',
                                     'I': '8',
                                     'J': '5',
                                     'Task Time Required': '4'}},
                         {'source_row': 5,
                          'values': {'A': '10',
                                     'B': '6',
                                     'C': '7',
                                     'D': '4',
                                     'E': '5',
                                     'F': '4',
                                     'G': '4',
                                     'H': '5',
                                     'I': '9',
                                     'J': '7',
                                     'Task Time Required': '5'}},
                         {'source_row': 6,
                          'values': {'A': '6',
                                     'B': '7',
                                     'C': '6',
                                     'D': '3',
                                     'E': '9',
                                     'F': '5',
                                     'G': '7',
                                     'H': '4',
                                     'I': '3',
                                     'J': '4',
                                     'Task Time Required': '6'}},
                         {'source_row': 7,
                          'values': {'A': '8',
                                     'B': '8',
                                     'C': '5',
                                     'D': '9',
                                     'E': '5',
                                     'F': '7',
                                     'G': '5',
                                     'H': '9',
                                     'I': '5',
                                     'J': '3',
                                     'Task Time Required': '7'}},
                         {'source_row': 8,
                          'values': {'A': '7',
                                     'B': '4',
                                     'C': '8',
                                     'D': '8',
                                     'E': '6',
                                     'F': '7',
                                     'G': '5',
                                     'H': '7',
                                     'I': '7',
                                     'J': '7',
                                     'Task Time Required': '8'}},
                         {'source_row': 9,
                          'values': {'A': '5',
                                     'B': '6',
                                     'C': '8',
                                     'D': '7',
                                     'E': '7',
                                     'F': '8',
                                     'G': '7',
                                     'H': '8',
                                     'I': '4',
                                     'J': '5',
                                     'Task Time Required': '9'}},
                         {'source_row': 10,
                          'values': {'A': '8',
                                     'B': '7',
                                     'C': '9',
                                     'D': '5',
                                     'E': '8',
                                     'F': '5',
                                     'G': '9',
                                     'H': '9',
                                     'I': '3',
                                     'J': '4',
                                     'Task Time Required': '10'}},
                         {'source_row': 11,
                          'values': {'A': '9',
                                     'B': '8',
                                     'C': '10',
                                     'D': '8',
                                     'E': '5',
                                     'F': '4',
                                     'G': '7',
                                     'H': '6',
                                     'I': '8',
                                     'J': '7',
                                     'Task Time Required': '11'}},
                         {'source_row': 12,
                          'values': {'A': '8',
                                     'B': '5',
                                     'C': '6',
                                     'D': '9',
                                     'E': '4',
                                     'F': '7',
                                     'G': '8',
                                     'H': '4',
                                     'I': '7',
                                     'J': '9',
                                     'Task Time Required': '12'}}],
             'returned_rows': 13,
             'role': 'assignment cost matrix',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
table = None
for t in data['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError('Assignment cost matrix table not found.')
columns = table['columns']
if 'Task Time Required' not in columns:
    raise ValueError("Missing 'Task Time Required' column in assignment cost matrix.")
workers = [col for col in columns if col != 'Task Time Required']
records = table['records']
tasks = []
for rec in records[1:]:
    t = rec['values']['Task Time Required']
    if t == '' or t is None:
        continue
    tasks.append(t)
if len(tasks) != 12:
    raise ValueError('Expected 12 tasks in data, found %d.' % len(tasks))
tasks = tasks[:10]
c = {}
for rec in records[1:]:
    t = rec['values']['Task Time Required']
    if t not in tasks:
        continue
    for w in workers:
        val = rec['values'][w]
        if val == '' or val is None:
            raise ValueError(f'Missing cost for worker {w}, task {t}.')
        try:
            c[w, t] = float(val)
        except Exception:
            raise ValueError(f'Non-numeric cost for worker {w}, task {t}: {val}')
if len(workers) != 10:
    raise ValueError(f'Expected 12 workers, found {len(workers)}. Workers: {workers}')
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(tasks)}. Tasks: {tasks}')
for w in workers:
    for t in tasks:
        if (w, t) not in c:
            raise ValueError(f'Missing cost for worker {w}, task {t}.')
m = gp.Model('worker_assignment')
m.Params.MIPGap = 0.0001
x = m.addVars(workers, tasks, vtype=GRB.BINARY, name='')
y = m.addVars(workers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[w, t] * x[w, t] for w in workers for t in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[w, t] for w in workers)) == 1 for t in tasks), name='')
m.addConstrs((gp.quicksum((x[w, t] for t in tasks)) <= y[w] for w in workers), name='')
m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
m.addConstrs((gp.quicksum((x[w, t] for t in tasks)) <= 1 for w in workers), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')