CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A maintenance company must assign a set of jobs to technician teams. The processing capacity of each team '
          'is listed in machine_capacity.csv. The cost of assigning each job to each team is provided in '
          'assignment_costs.csv, and the amount of team capacity consumed by each possible assignment is provided in '
          'assignment_resources.csv. Jobs are labeled J1 through J8 in the assignment data.\n'
          '\n'
          'Formulate a generalized assignment model. For each team i and job j, define x_ij as a binary variable equal '
          'to 1 if job j is assigned to team i and 0 otherwise. The objective is to minimize total assignment cost. '
          'The model should assign every job to exactly one team, ensure that the total capacity consumed on each team '
          'does not exceed its available capacity, and impose binary restrictions on all assignment variables.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Machine', 'Capacity'],
             'file_index': 0,
             'file_name': 'machine_capacity.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Capacity': '13', 'Machine': 'M1'}},
                         {'source_row': 1, 'values': {'Capacity': '12', 'Machine': 'M2'}},
                         {'source_row': 2, 'values': {'Capacity': '12', 'Machine': 'M3'}},
                         {'source_row': 3, 'values': {'Capacity': '12', 'Machine': 'M4'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Machine', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8'],
             'file_index': 1,
             'file_name': 'assignment_costs.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'J1': '8',
                                     'J2': '7',
                                     'J3': '25',
                                     'J4': '24',
                                     'J5': '27',
                                     'J6': '26',
                                     'J7': '28',
                                     'J8': '29',
                                     'Machine': 'M1'}},
                         {'source_row': 1,
                          'values': {'J1': '23',
                                     'J2': '24',
                                     'J3': '6',
                                     'J4': '9',
                                     'J5': '25',
                                     'J6': '27',
                                     'J7': '26',
                                     'J8': '28',
                                     'Machine': 'M2'}},
                         {'source_row': 2,
                          'values': {'J1': '27',
                                     'J2': '26',
                                     'J3': '24',
                                     'J4': '25',
                                     'J5': '5',
                                     'J6': '8',
                                     'J7': '23',
                                     'J8': '24',
                                     'Machine': 'M3'}},
                         {'source_row': 3,
                          'values': {'J1': '25',
                                     'J2': '27',
                                     'J3': '26',
                                     'J4': '24',
                                     'J5': '23',
                                     'J6': '25',
                                     'J7': '6',
                                     'J8': '7',
                                     'Machine': 'M4'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Machine', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8'],
             'file_index': 2,
             'file_name': 'assignment_resources.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'J1': '5',
                                     'J2': '6',
                                     'J3': '8',
                                     'J4': '7',
                                     'J5': '9',
                                     'J6': '8',
                                     'J7': '7',
                                     'J8': '7',
                                     'Machine': 'M1'}},
                         {'source_row': 1,
                          'values': {'J1': '8',
                                     'J2': '7',
                                     'J3': '4',
                                     'J4': '7',
                                     'J5': '8',
                                     'J6': '9',
                                     'J7': '8',
                                     'J8': '7',
                                     'Machine': 'M2'}},
                         {'source_row': 2,
                          'values': {'J1': '9',
                                     'J2': '8',
                                     'J3': '7',
                                     'J4': '8',
                                     'J5': '6',
                                     'J6': '5',
                                     'J7': '8',
                                     'J8': '7',
                                     'Machine': 'M3'}},
                         {'source_row': 3,
                          'values': {'J1': '8',
                                     'J2': '8',
                                     'J3': '7',
                                     'J4': '8',
                                     'J5': '8',
                                     'J6': '7',
                                     'J7': '5',
                                     'J8': '6',
                                     'Machine': 'M4'}}],
             'returned_rows': 4,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    cap_table = get_table('file_0_view_0')
    C_i = {}
    for rec in cap_table['records']:
        machine = rec['values']['Machine']
        capacity = int(rec['values']['Capacity'])
        C_i[machine] = capacity
    cost_table = get_table('file_1_view_0')
    c_ij = {}
    for rec in cost_table['records']:
        machine = rec['values']['Machine']
        for job in [f'J{i}' for i in range(1, 9)]:
            cost = int(rec['values'][job])
            c_ij[machine, job] = cost
    res_table = get_table('file_2_view_0')
    r_ij = {}
    for rec in res_table['records']:
        machine = rec['values']['Machine']
        for job in [f'J{i}' for i in range(1, 9)]:
            res = int(rec['values'][job])
            r_ij[machine, job] = res
    M = sorted(C_i.keys())
    J = [f'J{i}' for i in range(1, 9)]
    for m in M:
        if m not in C_i:
            raise ValueError(f'Missing capacity for machine {m}')
        for j in J:
            if (m, j) not in c_ij:
                raise ValueError(f'Missing cost for machine {m}, job {j}')
            if (m, j) not in r_ij:
                raise ValueError(f'Missing resource for machine {m}, job {j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in M for j in J], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((c_ij[i, j] * x[i, j] for i in M for j in J)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in M)) == 1 for j in J), name='')
    m.addConstrs((gurobipy.quicksum((r_ij[i, j] * x[i, j] for j in J)) <= C_i[i] for i in M), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()