CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are 40 tasks that must be run on 3 CPUs, with frequencies of 1.33, 2, and 2.66 GHz respectively (each '
          'processor can only run one task at a time). The basic instructions (in billions of instructions, BI) for '
          'each task are stored in 18.csv. Arrange the tasks onto the processors to minimize the completion time of '
          'the last task.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['1',
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
             'filters': {},
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
                                     '9': '4.4'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    if not table['records']:
        raise ValueError("No records found in table 'file_0_view_0'")
    record = table['records'][0]
    values = record['values']
    BI = {}
    for col in table['columns']:
        if col not in values:
            raise ValueError(f'Missing BI value for column {col}')
        try:
            BI[int(col)] = float(values[col])
        except ValueError:
            raise ValueError(f'Non-numeric BI value for column {col}: {values[col]}')
    T = list(range(1, 41))
    if set(BI.keys()) != set(T):
        missing = set(T) - set(BI.keys())
        raise ValueError(f'Missing BI values for tasks: {missing}')
    freq = {1: 1.33, 2: 2.0, 3: 2.66}
    M_set = [1, 2, 3]
    p = {}
    for i in T:
        for j in M_set:
            p[i, j] = BI[i] / freq[j]
    big_M = sum((max((p[i, j] for j in M_set)) for i in T))
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in T for j in M_set], vtype=GRB.BINARY, name='')
    y = m.addVars([(i, k, j) for i in T for k in T if i < k for j in M_set], vtype=GRB.BINARY, name='')
    s = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name='')
    c = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name='')
    Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='')
    for i in T:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in M_set)) == 1, name='assign_' + str(i))
    for i in T:
        for k in T:
            if i < k:
                for j in M_set:
                    m.addConstr(y[i, k, j] <= x[i, j], name=f'y_le_x_{i}_{k}_{j}_1')
                    m.addConstr(y[i, k, j] <= x[k, j], name=f'y_le_x_{i}_{k}_{j}_2')
                    m.addConstr(y[i, k, j] >= x[i, j] + x[k, j] - 1, name=f'y_ge_xsum_{i}_{k}_{j}')
    for i in T:
        for k in T:
            if i < k:
                for j in M_set:
                    m.addConstr(s[i] + p[i, j] <= s[k] + big_M * (1 - y[i, k, j]), name=f'seq1_{i}_{k}_{j}')
                    m.addConstr(s[k] + p[k, j] <= s[i] + big_M * y[i, k, j], name=f'seq2_{i}_{k}_{j}')
    for i in T:
        m.addConstr(c[i] == s[i] + gurobipy.quicksum((p[i, j] * x[i, j] for j in M_set)), name=f'c_def_{i}')
    for i in T:
        m.addConstr(c[i] <= Cmax, name=f'makespan_{i}')
    m.setObjective(Cmax, GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()