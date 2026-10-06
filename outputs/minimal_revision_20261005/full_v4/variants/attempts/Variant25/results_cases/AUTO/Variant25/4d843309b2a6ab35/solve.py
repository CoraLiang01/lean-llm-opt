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

def solve_problem(CSVQA_DATA):
    skill_level = {'junior': 1, 'intermediate': 2, 'senior': 3, 'expert': 4}
    workers_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            workers_table = t
            break
    if workers_table is None:
        raise RuntimeError('Workers table not found')
    workers_records = workers_table['records']
    worker_info = {}
    eligible_workers = []
    for rec in workers_records:
        vals = rec['values']
        wid = vals['worker_id']
        skill = vals['skill']
        on_leave = vals['on_leave']
        worker_info[wid] = {'skill': skill, 'on_leave': int(on_leave)}
        if int(on_leave) == 0:
            eligible_workers.append(wid)
    projects_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_1_view_0':
            projects_table = t
            break
    if projects_table is None:
        raise RuntimeError('Projects table not found')
    projects_records = projects_table['records']
    project_info = {}
    projects = []
    for rec in projects_records:
        vals = rec['values']
        pid = vals['project_id']
        required_skill = vals['required_skill']
        project_info[pid] = required_skill
        projects.append(pid)
    cost_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_2_view_0':
            cost_table = t
            break
    if cost_table is None:
        raise RuntimeError('Cost matrix table not found')
    cost_records = cost_table['records']
    cost_columns = [c for c in cost_table['columns'] if c != 'worker_id']
    cost = {}
    for rec in cost_records:
        vals = rec['values']
        wid = vals['worker_id']
        for pid in cost_columns:
            v = vals[pid]
            if v != '':
                try:
                    c = int(v)
                except Exception:
                    raise RuntimeError(f'Non-integer cost for ({wid},{pid}): {v}')
                cost[wid, pid] = c
    assignment_keys = []
    for wid in eligible_workers:
        w_skill = worker_info[wid]['skill']
        w_level = skill_level[w_skill.casefold()]
        for pid in projects:
            p_skill = project_info[pid]
            p_level = skill_level[p_skill.casefold()]
            if w_level >= p_level and (wid, pid) in cost:
                assignment_keys.append((wid, pid))
    for pid in projects:
        found = False
        for wid in eligible_workers:
            w_skill = worker_info[wid]['skill']
            w_level = skill_level[w_skill.casefold()]
            p_skill = project_info[pid]
            p_level = skill_level[p_skill.casefold()]
            if w_level >= p_level and (wid, pid) in cost:
                found = True
                break
        if not found:
            raise RuntimeError(f'No eligible worker for project {pid}')
    m = gp.Model('assignment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(assignment_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[wid, pid] * x[wid, pid] for (wid, pid) in assignment_keys)), GRB.MINIMIZE)
    for pid in projects:
        eligible_wids = [wid for wid in eligible_workers if (wid, pid) in cost and skill_level[worker_info[wid]['skill'].casefold()] >= skill_level[project_info[pid].casefold()]]
        m.addConstr(gp.quicksum((x[wid, pid] for wid in eligible_wids if (wid, pid) in x)) == 1, name='prj')
    for wid in eligible_workers:
        eligible_pids = [pid for pid in projects if (wid, pid) in cost and skill_level[worker_info[wid]['skill'].casefold()] >= skill_level[project_info[pid].casefold()]]
        m.addConstr(gp.quicksum((x[wid, pid] for pid in eligible_pids if (wid, pid) in x)) <= 1, name='wrk')
    m.optimize()
    return m
m = solve_problem(CSVQA_DATA)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')