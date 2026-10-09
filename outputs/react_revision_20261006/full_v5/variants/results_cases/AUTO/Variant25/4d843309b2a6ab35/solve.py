CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Assign six construction projects to the eight listed managers. Each project needs exactly one person, and '
          'each person may take at most one project. Exclude anyone with on_leave=1. The assigned person must meet '
          'required_skill, with Junior < Intermediate < Senior < Expert. In the cost matrix, project IDs are columns '
          'and worker IDs identify rows. A blank cell forbids that assignment. Minimize the sum of assignment costs, '
          'and report the minimum in USD cents.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['worker_id', 'skill', 'on_leave'],
             'file_index': 0,
             'file_name': 'export_01.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W12'}},
                         {'source_row': 1, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W06'}},
                         {'source_row': 2, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W04'}},
                         {'source_row': 3, 'values': {'on_leave': '1', 'skill': 'Junior', 'worker_id': 'W01'}},
                         {'source_row': 4, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W11'}},
                         {'source_row': 5, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W10'}},
                         {'source_row': 6, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W02'}},
                         {'source_row': 7, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W00'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['project_id', 'required_skill'],
             'file_index': 1,
             'file_name': 'export_02.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'project_id': 'P00', 'required_skill': 'Junior'}},
                         {'source_row': 1, 'values': {'project_id': 'P01', 'required_skill': 'Junior'}},
                         {'source_row': 2, 'values': {'project_id': 'P02', 'required_skill': 'Junior'}},
                         {'source_row': 3, 'values': {'project_id': 'P03', 'required_skill': 'Junior'}},
                         {'source_row': 4, 'values': {'project_id': 'P04', 'required_skill': 'Junior'}},
                         {'source_row': 5, 'values': {'project_id': 'P05', 'required_skill': 'Intermediate'}}],
             'returned_rows': 6,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['worker_id', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05'],
             'file_index': 2,
             'file_name': 'export_03.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'P00': '1091',
                                     'P01': '324',
                                     'P02': '379',
                                     'P03': '272',
                                     'P04': '',
                                     'P05': '',
                                     'worker_id': 'W11'}},
                         {'source_row': 1,
                          'values': {'P00': '1176',
                                     'P01': '1111',
                                     'P02': '1380',
                                     'P03': '542',
                                     'P04': '158',
                                     'P05': '922',
                                     'worker_id': 'W10'}},
                         {'source_row': 2,
                          'values': {'P00': '',
                                     'P01': '822',
                                     'P02': '',
                                     'P03': '223',
                                     'P04': '1155',
                                     'P05': '1055',
                                     'worker_id': 'W06'}},
                         {'source_row': 3,
                          'values': {'P00': '5',
                                     'P01': '',
                                     'P02': '20',
                                     'P03': '',
                                     'P04': '18',
                                     'P05': '7',
                                     'worker_id': 'W01'}},
                         {'source_row': 4,
                          'values': {'P00': '',
                                     'P01': '1397',
                                     'P02': '953',
                                     'P03': '714',
                                     'P04': '205',
                                     'P05': '',
                                     'worker_id': 'W02'}},
                         {'source_row': 5,
                          'values': {'P00': '1063',
                                     'P01': '219',
                                     'P02': '',
                                     'P03': '1329',
                                     'P04': '',
                                     'P05': '436',
                                     'worker_id': 'W00'}},
                         {'source_row': 6,
                          'values': {'P00': '642',
                                     'P01': '',
                                     'P02': '1130',
                                     'P03': '133',
                                     'P04': '199',
                                     'P05': '311',
                                     'worker_id': 'W04'}},
                         {'source_row': 7,
                          'values': {'P00': '102',
                                     'P01': '353',
                                     'P02': '651',
                                     'P03': '102',
                                     'P04': '',
                                     'P05': '7',
                                     'worker_id': 'W12'}}],
             'returned_rows': 8,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [8, 6], "
                                   "'expected_shape': [7, 6], 'row_ids_aligned': False, 'column_ids_aligned': True, "
                                   "'row_mapping_basis': 'unresolved', 'column_mapping_basis': 'exact'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [8, 6], "
                                   "'expected_shape': [7, 6], 'row_ids_aligned': False, 'column_ids_aligned': True, "
                                   "'row_mapping_basis': 'unresolved', 'column_mapping_basis': 'exact'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    skill_order = {'junior': 0, 'intermediate': 1, 'senior': 2, 'expert': 3}
    workers_data = [rec['values'] for rec in CSVQA_DATA['tables'][0]['records']]
    eligible_workers = [w['worker_id'] for w in workers_data if w['on_leave'] == '0']
    worker_skill = {w['worker_id']: w['skill'] for w in workers_data if w['on_leave'] == '0'}
    projects_data = [rec['values'] for rec in CSVQA_DATA['tables'][1]['records']]
    projects = [p['project_id'] for p in projects_data]
    project_required_skill = {p['project_id']: p['required_skill'] for p in projects_data}
    cost_table = CSVQA_DATA['tables'][2]
    cost_records = cost_table['records']
    cost_matrix = {}
    for rec in cost_records:
        row = rec['values']
        wid = row['worker_id']
        cost_matrix[wid] = {}
        for pid in projects:
            val = row[pid]
            cost_matrix[wid][pid] = int(val) if val != '' else None
    eligible_pairs = []
    cost = {}
    for w in eligible_workers:
        cost[w] = {}
        w_skill = worker_skill[w].casefold()
        w_skill_level = skill_order[w_skill]
        for p in projects:
            p_skill = project_required_skill[p].casefold()
            p_skill_level = skill_order[p_skill]
            c = cost_matrix[w][p] if w in cost_matrix and p in cost_matrix[w] else None
            if c is not None and w_skill_level >= p_skill_level:
                eligible_pairs.append((w, p))
                cost[w][p] = c
            else:
                cost[w][p] = None
    for (w, p) in eligible_pairs:
        if cost[w][p] is None:
            raise ValueError(f'Missing cost for eligible assignment ({w}, {p})')
    m = gp.Model('assignment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w][p] * x[w, p] for (w, p) in eligible_pairs)), GRB.MINIMIZE)
    for p in projects:
        m.addConstr(gp.quicksum((x[w, p] for w in eligible_workers if (w, p) in eligible_pairs)) == 1, name='proj')
    for w in eligible_workers:
        m.addConstr(gp.quicksum((x[w, p] for p in projects if (w, p) in eligible_pairs)) <= 1, name='work')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')