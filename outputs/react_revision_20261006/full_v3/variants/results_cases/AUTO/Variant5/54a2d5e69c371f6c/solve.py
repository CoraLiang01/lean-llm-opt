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
 'route': 'AP',
 'tables': [{'columns': ['Machine', 'Capacity'],
             'file_index': 0,
             'file_name': 'machine_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0, 'values': {'Capacity': '13', 'Machine': 'M1'}},
                         {'source_row': 1, 'values': {'Capacity': '12', 'Machine': 'M2'}},
                         {'source_row': 2, 'values': {'Capacity': '12', 'Machine': 'M3'}},
                         {'source_row': 3, 'values': {'Capacity': '12', 'Machine': 'M4'}}],
             'returned_rows': 4,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Machine', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8'],
             'file_index': 1,
             'file_name': 'assignment_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Machine', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8'],
             'file_index': 2,
             'file_name': 'assignment_resources.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_1_view_0', 'shape': [4, 8], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_1_view_0', 'shape': [4, 8], "
                                   "'expected_shape': [4, 4], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    cap_table = tables['file_0_view_0']
    cost_table = tables['file_1_view_0']
    res_table = tables['file_2_view_0']
    machines = [rec['values']['Machine'] for rec in cap_table['records']]
    jobs = [col for col in cost_table['columns'] if col != 'Machine']
    cap = {}
    for rec in cap_table['records']:
        machine = rec['values']['Machine']
        cap[machine] = float(rec['values']['Capacity'])
    cost = {}
    for rec in cost_table['records']:
        machine = rec['values']['Machine']
        cost[machine] = {}
        for job in jobs:
            val = rec['values'][job]
            if val is None or val == '':
                raise ValueError(f'Missing cost for ({machine},{job})')
            cost[machine][job] = float(val)
    res = {}
    for rec in res_table['records']:
        machine = rec['values']['Machine']
        res[machine] = {}
        for job in jobs:
            val = rec['values'][job]
            if val is None or val == '':
                raise ValueError(f'Missing resource for ({machine},{job})')
            res[machine][job] = float(val)
    for machine in machines:
        if machine not in cost or machine not in res or machine not in cap:
            raise ValueError(f'Missing data for machine {machine}')
        for job in jobs:
            if job not in cost[machine] or job not in res[machine]:
                raise ValueError(f'Missing data for ({machine},{job})')
    m = gp.Model('GeneralizedAssignment')
    m.setParam('MIPGap', 0.0001)
    x_keys = [(i, j) for i in machines for j in jobs]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in machines for j in jobs)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in machines)) == 1 for j in jobs), name='')
    m.addConstrs((gp.quicksum((res[i][j] * x[i, j] for j in jobs)) <= cap[i] for i in machines), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_DATA)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')