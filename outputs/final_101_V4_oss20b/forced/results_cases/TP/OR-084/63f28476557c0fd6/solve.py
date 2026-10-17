CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are 40 tasks that must be run on 3 CPUs, with frequencies of 1.33, 2, and 2.66 GHz respectively (each '
          'processor can only run one task at a time). The basic instructions (in billions of instructions, BI) for '
          'each task are stored in 18.csv. Arrange the tasks onto the processors to minimize the completion time of '
          'the last task.',
 'relationships': [],
 'route': 'TP',
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
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    data_table = next((t for t in tables if 'records' in t))
    records = data_table['records']
    if not records:
        raise ValueError('No records found in data table.')
    record = records[0]
    values = record['values']
    BI = {}
    for col, val in values.items():
        if col == 'Process':
            continue
        try:
            task_id = int(col)
        except ValueError:
            continue
        try:
            BI[task_id] = float(val)
        except ValueError:
            raise ValueError(f'Invalid BI value for task {task_id}: {val}')
    T = list(range(1, 41))
    missing_tasks = [i for i in T if i not in BI]
    if missing_tasks:
        raise ValueError(f'Missing BI values for tasks: {missing_tasks}')
    f = {1: 1.33, 2: 2.0, 3: 2.66}
    M = [1, 2, 3]
    missing_freq = [k for k in M if k not in f]
    if missing_freq:
        raise ValueError(f'Missing frequency values for CPUs: {missing_freq}')
    total_BI = sum((BI[i] for i in T))
    min_f = min((f[k] for k in M))
    M_big = total_BI / min_f
    m = gurobipy.Model('TaskScheduling')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, k) for i in T for k in M], vtype=GRB.BINARY, name='')
    s = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name='')
    C = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name='')
    C_max = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars([(i, j, k) for i in T for j in T if i < j for k in M], vtype=GRB.BINARY, name='')
    for i in T:
        m.addConstr(gurobipy.quicksum((x[i, k] for k in M)) == 1, name=f'assign_{i}')
    for i in T:
        p_sum = gurobipy.quicksum((x[i, k] * (BI[i] / f[k]) for k in M))
        m.addConstr(C[i] == s[i] + p_sum, name=f'comp_{i}')
    for i in T:
        m.addConstr(C_max >= C[i], name=f'makespan_{i}')
    for i in T:
        for j in T:
            if i >= j:
                continue
            for k in M:
                p_i_k = BI[i] / f[k]
                p_j_k = BI[j] / f[k]
                m.addConstr(s[i] + p_i_k <= s[j] + M_big * (1 - y[i, j, k]) + M_big * (2 - x[i, k] - x[j, k]), name=f'big1_{i}_{j}_{k}')
                m.addConstr(s[j] + p_j_k <= s[i] + M_big * y[i, j, k] + M_big * (2 - x[i, k] - x[j, k]), name=f'big2_{i}_{j}_{k}')
    m.setObjective(C_max, GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()