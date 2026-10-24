CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Now we need to determine 10 out of 12 workers to each complete one of the ten tasks. Since each worker has '
          'different skill sets, the amount of time required for each worker to complete each task is also different. '
          'The time required for each worker to complete each task is shown in 15.csv.\n'
          '\n'
          '    Try to find a work assignment plan that minimizes the total working hours.',
 'relationships': [],
 'route': 'AP',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError('Required table not found')
    records = table['records']
    columns = table['columns']
    worker_cols = [col for col in columns if col != 'Task Time Required']
    W = worker_cols
    T = []
    time = {}
    for rec in records:
        source_row = rec.get('source_row')
        if source_row is None:
            continue
        values = rec['values']
        task_val = values.get('Task Time Required')
        if task_val is None:
            continue
        if source_row == 0:
            continue
        try:
            t = int(task_val)
        except ValueError:
            continue
        if t < 1 or t > 10:
            continue
        T.append(t)
        for w in W:
            val_str = values.get(w, '')
            if val_str == '':
                raise ValueError(f'Missing time for worker {w} task {t}')
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Invalid time value for worker {w} task {t}')
            time[w, t] = val
    if set(T) != set(range(1, 11)):
        missing = set(range(1, 11)) - set(T)
        raise ValueError(f'Missing task rows: {missing}')
    for w in W:
        for t in T:
            if (w, t) not in time:
                raise ValueError(f'Missing time for worker {w} task {t}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(w, t) for w in W for t in T], vtype=GRB.BINARY, name='')
    obj_expr = gurobipy.quicksum((time[w, t] * x[w, t] for w in W for t in T))
    model.setObjective(obj_expr, GRB.MINIMIZE)
    for t in T:
        model.addConstr(gurobipy.quicksum((x[w, t] for w in W)) == 1)
    for w in W:
        model.addConstr(gurobipy.quicksum((x[w, t] for t in T)) <= 1)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(model.ObjVal)
        for var in model.getVars():
            print(var.VarName, var.X)
    else:
        print(model.Status)
    return model
m = solve_problem()