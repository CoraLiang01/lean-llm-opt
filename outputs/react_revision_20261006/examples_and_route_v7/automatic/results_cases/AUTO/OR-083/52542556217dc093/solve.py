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
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'worker-task time matrix',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    import pandas as pd
    worker_set = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    task_set = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    table_id = 'file_0_view_0'
    df = CSVQA_FRAMES[table_id]
    header_row = df.iloc[0]
    if not all(header_row[task_set].values == task_set):
        raise ValueError('Task columns do not match expected task_set')
    t = {}
    for (idx, worker) in enumerate(worker_set):
        row = df.iloc[idx + 1]
        if str(row['Task Time Required']) != worker:
            raise ValueError(f"Worker row mismatch at index {idx + 1}: expected {worker}, got {row['Task Time Required']}")
        for task in task_set:
            val = row[task]
            try:
                t[worker, task] = int(val)
            except Exception:
                raise ValueError(f'Invalid or missing time value for worker {worker}, task {task}: {val}')
    m = gp.Model('worker_task_assignment')
    x_vars = m.addVars(worker_set, task_set, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((t[i, j] * x_vars[i, j] for i in worker_set for j in task_set)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in worker_set)) == 1 for j in task_set), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in task_set)) <= 1 for i in worker_set), name='')
    m.addConstr(gp.quicksum((x_vars[i, j] for i in worker_set for j in task_set)) == 10, name='total_assign')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()