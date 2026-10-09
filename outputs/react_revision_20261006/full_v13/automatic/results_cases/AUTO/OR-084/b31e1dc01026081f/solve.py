CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are 40 tasks that must be run on 3 CPUs, with frequencies of 1.33, 2, and 2.66 GHz respectively (each '
          'processor can only run one task at a time). The basic instructions (in billions of instructions, BI) for '
          'each task are stored in 18.csv. Arrange the tasks onto the processors to minimize the completion time of '
          'the last task.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Process',
                         '1',
                         '2',
                         '3',
                         '4',
                         '5',
                         '6',
                         '7',
                         '8',
                         '9',
                         '10',
                         '11',
                         '12',
                         '13',
                         '14',
                         '15',
                         '16',
                         '17',
                         '18',
                         '19',
                         '20',
                         '21',
                         '22',
                         '23',
                         '24',
                         '25',
                         '26',
                         '27',
                         '28',
                         '29',
                         '30',
                         '31',
                         '32',
                         '33',
                         '34',
                         '35',
                         '36',
                         '37',
                         '38',
                         '39',
                         '40'],
             'file_index': 0,
             'file_name': '18.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0,
                          'values': {'1': '1.1',
                                     '10': '3.8',
                                     '11': '3.5',
                                     '12': '2.8',
                                     '13': '4.1',
                                     '14': '2.9',
                                     '15': '5.4',
                                     '16': '5.8',
                                     '17': '2.6',
                                     '18': '4.9',
                                     '19': '3.4',
                                     '2': '2.1',
                                     '20': '3.6',
                                     '21': '5.6',
                                     '22': '0.9',
                                     '23': '1',
                                     '24': '0.6',
                                     '25': '5.1',
                                     '26': '4.8',
                                     '27': '5.3',
                                     '28': '5.9',
                                     '29': '4.9',
                                     '3': '3',
                                     '30': '3',
                                     '31': '4.8',
                                     '32': '1.2',
                                     '33': '4',
                                     '34': '1.3',
                                     '35': '5.7',
                                     '36': '3.4',
                                     '37': '2.8',
                                     '38': '2',
                                     '39': '4.8',
                                     '4': '1',
                                     '40': '3',
                                     '5': '0.7',
                                     '6': '5',
                                     '7': '3',
                                     '8': '3.5',
                                     '9': '4.4',
                                     'Process': 'BI'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    table_id = 'file_0_view_0'
    frame = CSVQA_FRAMES[table_id]
    task_ids = [str(i) for i in range(1, 41)]
    proc_ids = [1, 2, 3]
    freq = {1: 1.33, 2: 2.0, 3: 2.66}
    bi_row = None
    for (idx, row) in frame.iterrows():
        if str(row['Process']).casefold() == 'bi':
            bi_row = row
            break
    if bi_row is None:
        raise ValueError("No row with Process == 'BI' found in 18.csv")
    b_t = {}
    for t in task_ids:
        val = bi_row[t]
        try:
            b_t[t] = float(val)
        except Exception:
            raise ValueError(f'Invalid BI value for task {t}: {val}')
    if set(b_t.keys()) != set(task_ids):
        raise ValueError('Missing BI values for some tasks')
    m = gp.Model('TaskAssignment')
    x_vars = m.addVars(task_ids, proc_ids, vtype=gp.GRB.BINARY, name='')
    Cmax = m.addVar(lb=0.0, name='Cmax')
    m.setObjective(Cmax, gp.GRB.MINIMIZE)
    for t in task_ids:
        m.addConstr(gp.quicksum((x_vars[t, p] for p in proc_ids)) == 1, name=f'Assign_{t}')
    for p in proc_ids:
        m.addConstr(gp.quicksum((b_t[t] / freq[p] * x_vars[t, p] for t in task_ids)) <= Cmax, name=f'ProcTime_{p}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)