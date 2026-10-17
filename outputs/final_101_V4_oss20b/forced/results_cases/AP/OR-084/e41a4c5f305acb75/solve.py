CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are 40 tasks that must be run on 3 CPUs, with frequencies of 1.33, 2, and 2.66 GHz respectively (each '
          'processor can only run one task at a time). The basic instructions (in billions of instructions, BI) for '
          'each task are stored in 18.csv. Arrange the tasks onto the processors to minimize the completion time of '
          'the last task.',
 'relationships': [],
 'route': 'AP',
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
    table = CSVQA_DATA['tables'][0]
    record = table['records'][0]['values']
    BI = {}
    for col, val in record.items():
        if col.isdigit():
            BI[int(col)] = float(val)
    T = list(range(1, 41))
    M_set = [1, 2, 3]
    f = {1: 1.33, 2: 2.0, 3: 2.66}
    p = {(i, j): BI[j] / f[i] for i in M_set for j in T}
    M_big = sum(BI.values()) / min(f.values())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in M_set for j in T], vtype=GRB.BINARY, name='')
    s = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name='')
    C = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name='')
    Cmax = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Cmax')
    y = m.addVars([(j, k) for j in T for k in T if j < k], vtype=GRB.BINARY, name='')
    m.setObjective(Cmax, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in M_set)) == 1 for j in T), name='')
    m.addConstrs((C[j] - s[j] - gurobipy.quicksum((p[i, j] * x[i, j] for i in M_set)) == 0 for j in T), name='')
    for j in T:
        for k in T:
            if j < k:
                m.addConstr(s[k] >= C[j] - M_big * (1 - y[j, k]), name=f'seq1_{j}_{k}')
                m.addConstr(s[j] >= C[k] - M_big * y[j, k], name=f'seq2_{j}_{k}')
    m.addConstrs((C[j] <= Cmax for j in T), name='')
    m.optimize()
    return m
m = solve_problem()