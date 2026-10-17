CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'There are 40 tasks that must be run on 3 CPUs, with frequencies of 1.33, 2, and 2.66 GHz respectively (each '
          'processor can only run one task at a time). The basic instructions (in billions of instructions, BI) for '
          'each task are stored in 18.csv. Arrange the tasks onto the processors to minimize the completion time of '
          'the last task.',
 'relationships': [],
 'route': 'RA',
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
    I = list(range(1, 41))
    J = [1, 2, 3]
    f = {1: 1.33, 2: 2.0, 3: 2.66}
    BI = {}
    for i in I:
        key = str(i)
        if key not in record:
            raise ValueError(f'Missing BI value for task {i}')
        BI[i] = float(record[key])
    P = {}
    for i in I:
        for j in J:
            P[i, j] = BI[i] / f[j]
    M = sum((max((P[i, j] for j in J)) for i in I))
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.BINARY, name='')
    s = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    C = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    z = m.addVars([(i, k, j) for i in I for k in I if i < k for j in J], vtype=GRB.BINARY, name='')
    T = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='T')
    m.setObjective(T, GRB.MINIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in J)) == 1, name=f'assign_{i}')
    for i in I:
        m.addConstr(C[i] >= s[i] + gurobipy.quicksum((P[i, j] * x[i, j] for j in J)), name=f'comp_{i}')
    for i in I:
        m.addConstr(T >= C[i], name=f'makespan_{i}')
    for i in I:
        for k in I:
            if i >= k:
                continue
            for j in J:
                m.addConstr(s[k] >= s[i] + P[i, j] - M * (1 - z[i, k, j]) - M * (1 - x[i, j]) - M * (1 - x[k, j]), name=f'nonoverlap1_{i}_{k}_{j}')
                m.addConstr(s[i] >= s[k] + P[k, j] - M * z[i, k, j] - M * (1 - x[i, j]) - M * (1 - x[k, j]), name=f'nonoverlap2_{i}_{k}_{j}')
                m.addConstr(z[i, k, j] + z[k, i, j] == 1, name=f'zpair_{i}_{k}_{j}')
                m.addConstr(z[i, k, j] <= x[i, j], name=f'zxij_{i}_{k}_{j}')
                m.addConstr(z[i, k, j] <= x[k, j], name=f'zxkj_{i}_{k}_{j}')
                m.addConstr(z[k, i, j] <= x[k, j], name=f'zxkj2_{i}_{k}_{j}')
                m.addConstr(z[k, i, j] <= x[i, j], name=f'zxij2_{i}_{k}_{j}')
    m.optimize()
    return m
m = solve_problem()