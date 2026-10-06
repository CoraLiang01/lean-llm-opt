CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A service team must cover eight jobs. The offer list gives cost_cents for each available '
          'worker_id/project_id pairing among ten staff members. Each project needs exactly one person, and each '
          'person may take at most one project. Exclude anyone with on_leave=1. The assigned person must meet '
          'required_skill, with Junior < Intermediate < Senior < Expert. Only listed offers are permitted. Minimize '
          'the sum of assignment costs, and report the minimum in USD cents.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['worker_id', 'skill', 'on_leave'],
             'file_index': 0,
             'file_name': 'export_01.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'on_leave': '1', 'skill': 'Senior', 'worker_id': 'W01'}},
                         {'source_row': 1, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W00'}},
                         {'source_row': 2, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W19'}},
                         {'source_row': 3, 'values': {'on_leave': '0', 'skill': 'Senior', 'worker_id': 'W11'}},
                         {'source_row': 4, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W04'}},
                         {'source_row': 5, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W15'}},
                         {'source_row': 6, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W10'}},
                         {'source_row': 7, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W06'}},
                         {'source_row': 8, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W20'}},
                         {'source_row': 9, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W14'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['project_id', 'required_skill'],
             'file_index': 1,
             'file_name': 'export_02.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'project_id': 'P00', 'required_skill': 'Intermediate'}},
                         {'source_row': 1, 'values': {'project_id': 'P01', 'required_skill': 'Intermediate'}},
                         {'source_row': 2, 'values': {'project_id': 'P02', 'required_skill': 'Senior'}},
                         {'source_row': 3, 'values': {'project_id': 'P03', 'required_skill': 'Junior'}},
                         {'source_row': 4, 'values': {'project_id': 'P04', 'required_skill': 'Junior'}},
                         {'source_row': 5, 'values': {'project_id': 'P05', 'required_skill': 'Junior'}},
                         {'source_row': 6, 'values': {'project_id': 'P06', 'required_skill': 'Junior'}},
                         {'source_row': 7, 'values': {'project_id': 'P07', 'required_skill': 'Junior'}}],
             'returned_rows': 8,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['worker_id', 'project_id', 'cost_cents'],
             'file_index': 2,
             'file_name': 'export_03.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 62,
             'records': [{'source_row': 0, 'values': {'cost_cents': '147', 'project_id': 'P00', 'worker_id': 'W10'}},
                         {'source_row': 1, 'values': {'cost_cents': '114', 'project_id': 'P01', 'worker_id': 'W10'}},
                         {'source_row': 2, 'values': {'cost_cents': '998', 'project_id': 'P03', 'worker_id': 'W10'}},
                         {'source_row': 3, 'values': {'cost_cents': '1270', 'project_id': 'P04', 'worker_id': 'W10'}},
                         {'source_row': 4, 'values': {'cost_cents': '953', 'project_id': 'P07', 'worker_id': 'W10'}},
                         {'source_row': 5, 'values': {'cost_cents': '5', 'project_id': 'P00', 'worker_id': 'W19'}},
                         {'source_row': 6, 'values': {'cost_cents': '9', 'project_id': 'P02', 'worker_id': 'W19'}},
                         {'source_row': 7, 'values': {'cost_cents': '109', 'project_id': 'P03', 'worker_id': 'W19'}},
                         {'source_row': 8, 'values': {'cost_cents': '1306', 'project_id': 'P04', 'worker_id': 'W19'}},
                         {'source_row': 9, 'values': {'cost_cents': '626', 'project_id': 'P05', 'worker_id': 'W19'}},
                         {'source_row': 10, 'values': {'cost_cents': '466', 'project_id': 'P06', 'worker_id': 'W19'}},
                         {'source_row': 11, 'values': {'cost_cents': '1365', 'project_id': 'P07', 'worker_id': 'W19'}},
                         {'source_row': 12, 'values': {'cost_cents': '235', 'project_id': 'P00', 'worker_id': 'W11'}},
                         {'source_row': 13, 'values': {'cost_cents': '1034', 'project_id': 'P02', 'worker_id': 'W11'}},
                         {'source_row': 14, 'values': {'cost_cents': '782', 'project_id': 'P03', 'worker_id': 'W11'}},
                         {'source_row': 15, 'values': {'cost_cents': '556', 'project_id': 'P04', 'worker_id': 'W11'}},
                         {'source_row': 16, 'values': {'cost_cents': '1018', 'project_id': 'P05', 'worker_id': 'W11'}},
                         {'source_row': 17, 'values': {'cost_cents': '1136', 'project_id': 'P07', 'worker_id': 'W11'}},
                         {'source_row': 18, 'values': {'cost_cents': '17', 'project_id': 'P01', 'worker_id': 'W00'}},
                         {'source_row': 19, 'values': {'cost_cents': '430', 'project_id': 'P04', 'worker_id': 'W00'}},
                         {'source_row': 20, 'values': {'cost_cents': '1385', 'project_id': 'P05', 'worker_id': 'W00'}},
                         {'source_row': 21, 'values': {'cost_cents': '260', 'project_id': 'P06', 'worker_id': 'W00'}},
                         {'source_row': 22, 'values': {'cost_cents': '1107', 'project_id': 'P07', 'worker_id': 'W00'}},
                         {'source_row': 23, 'values': {'cost_cents': '675', 'project_id': 'P00', 'worker_id': 'W20'}},
                         {'source_row': 24, 'values': {'cost_cents': '1055', 'project_id': 'P01', 'worker_id': 'W20'}},
                         {'source_row': 25, 'values': {'cost_cents': '8', 'project_id': 'P02', 'worker_id': 'W20'}},
                         {'source_row': 26, 'values': {'cost_cents': '703', 'project_id': 'P03', 'worker_id': 'W20'}},
                         {'source_row': 27, 'values': {'cost_cents': '893', 'project_id': 'P06', 'worker_id': 'W20'}},
                         {'source_row': 28, 'values': {'cost_cents': '129', 'project_id': 'P07', 'worker_id': 'W20'}},
                         {'source_row': 29, 'values': {'cost_cents': '4', 'project_id': 'P00', 'worker_id': 'W04'}},
                         {'source_row': 30, 'values': {'cost_cents': '8', 'project_id': 'P01', 'worker_id': 'W04'}},
                         {'source_row': 31, 'values': {'cost_cents': '16', 'project_id': 'P02', 'worker_id': 'W04'}},
                         {'source_row': 32, 'values': {'cost_cents': '543', 'project_id': 'P03', 'worker_id': 'W04'}},
                         {'source_row': 33, 'values': {'cost_cents': '205', 'project_id': 'P04', 'worker_id': 'W04'}},
                         {'source_row': 34, 'values': {'cost_cents': '1149', 'project_id': 'P05', 'worker_id': 'W04'}},
                         {'source_row': 35, 'values': {'cost_cents': '533', 'project_id': 'P06', 'worker_id': 'W04'}},
                         {'source_row': 36, 'values': {'cost_cents': '732', 'project_id': 'P07', 'worker_id': 'W04'}},
                         {'source_row': 37, 'values': {'cost_cents': '19', 'project_id': 'P00', 'worker_id': 'W01'}},
                         {'source_row': 38, 'values': {'cost_cents': '6', 'project_id': 'P02', 'worker_id': 'W01'}},
                         {'source_row': 39, 'values': {'cost_cents': '19', 'project_id': 'P05', 'worker_id': 'W01'}},
                         {'source_row': 40, 'values': {'cost_cents': '8', 'project_id': 'P06', 'worker_id': 'W01'}},
                         {'source_row': 41, 'values': {'cost_cents': '13', 'project_id': 'P07', 'worker_id': 'W01'}},
                         {'source_row': 42, 'values': {'cost_cents': '1217', 'project_id': 'P00', 'worker_id': 'W06'}},
                         {'source_row': 43, 'values': {'cost_cents': '425', 'project_id': 'P01', 'worker_id': 'W06'}},
                         {'source_row': 44, 'values': {'cost_cents': '1320', 'project_id': 'P02', 'worker_id': 'W06'}},
                         {'source_row': 45, 'values': {'cost_cents': '1097', 'project_id': 'P03', 'worker_id': 'W06'}},
                         {'source_row': 46, 'values': {'cost_cents': '822', 'project_id': 'P04', 'worker_id': 'W06'}},
                         {'source_row': 47, 'values': {'cost_cents': '221', 'project_id': 'P05', 'worker_id': 'W06'}},
                         {'source_row': 48, 'values': {'cost_cents': '496', 'project_id': 'P06', 'worker_id': 'W06'}},
                         {'source_row': 49, 'values': {'cost_cents': '908', 'project_id': 'P07', 'worker_id': 'W06'}},
                         {'source_row': 50, 'values': {'cost_cents': '1361', 'project_id': 'P01', 'worker_id': 'W14'}},
                         {'source_row': 51, 'values': {'cost_cents': '379', 'project_id': 'P03', 'worker_id': 'W14'}},
                         {'source_row': 52, 'values': {'cost_cents': '239', 'project_id': 'P04', 'worker_id': 'W14'}},
                         {'source_row': 53, 'values': {'cost_cents': '460', 'project_id': 'P05', 'worker_id': 'W14'}},
                         {'source_row': 54, 'values': {'cost_cents': '130', 'project_id': 'P06', 'worker_id': 'W14'}},
                         {'source_row': 55, 'values': {'cost_cents': '722', 'project_id': 'P00', 'worker_id': 'W15'}},
                         {'source_row': 56, 'values': {'cost_cents': '577', 'project_id': 'P02', 'worker_id': 'W15'}},
                         {'source_row': 57, 'values': {'cost_cents': '1062', 'project_id': 'P03', 'worker_id': 'W15'}},
                         {'source_row': 58, 'values': {'cost_cents': '1314', 'project_id': 'P04', 'worker_id': 'W15'}},
                         {'source_row': 59, 'values': {'cost_cents': '528', 'project_id': 'P05', 'worker_id': 'W15'}},
                         {'source_row': 60, 'values': {'cost_cents': '835', 'project_id': 'P06', 'worker_id': 'W15'}},
                         {'source_row': 61, 'values': {'cost_cents': '918', 'project_id': 'P07', 'worker_id': 'W15'}}],
             'returned_rows': 62,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix source IDs must be non-empty and unique: ['W10', 'W10', 'W10', 'W10', "
                                   "'W10', 'W19', 'W19', 'W19', 'W19', 'W19', 'W19', 'W19', 'W11', 'W11', 'W11', "
                                   "'W11', 'W11', 'W11', 'W00', 'W00', 'W00', 'W00', 'W00', 'W20', 'W20', 'W20', "
                                   "'W20', 'W20', 'W20', 'W04', 'W04', 'W04', 'W04', 'W04', 'W04', 'W04', 'W04', "
                                   "'W01', 'W01', 'W01', 'W01', 'W01', 'W06', 'W06', 'W06', 'W06', 'W06', 'W06', "
                                   "'W06', 'W06', 'W14', 'W14', 'W14', 'W14', 'W14', 'W15', 'W15', 'W15', 'W15', "
                                   "'W15', 'W15', 'W15']",
                'planner_errors': ["Matrix source IDs must be non-empty and unique: ['W10', 'W10', 'W10', 'W10', "
                                   "'W10', 'W19', 'W19', 'W19', 'W19', 'W19', 'W19', 'W19', 'W11', 'W11', 'W11', "
                                   "'W11', 'W11', 'W11', 'W00', 'W00', 'W00', 'W00', 'W00', 'W20', 'W20', 'W20', "
                                   "'W20', 'W20', 'W20', 'W04', 'W04', 'W04', 'W04', 'W04', 'W04', 'W04', 'W04', "
                                   "'W01', 'W01', 'W01', 'W01', 'W01', 'W06', 'W06', 'W06', 'W06', 'W06', 'W06', "
                                   "'W06', 'W06', 'W14', 'W14', 'W14', 'W14', 'W14', 'W15', 'W15', 'W15', 'W15', "
                                   "'W15', 'W15', 'W15']"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    skill_order = {'junior': 0, 'intermediate': 1, 'senior': 2, 'expert': 3}
    workers_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            workers_table = t
            break
    if workers_table is None:
        raise RuntimeError('Missing workers table')
    eligible_workers = []
    worker_skill = {}
    for rec in workers_table['records']:
        wid = rec['values']['worker_id']
        on_leave = rec['values']['on_leave']
        skill = rec['values']['skill']
        if on_leave == '0':
            eligible_workers.append(wid)
            worker_skill[wid] = skill
    projects_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_1_view_0':
            projects_table = t
            break
    if projects_table is None:
        raise RuntimeError('Missing projects table')
    projects = []
    project_required_skill = {}
    for rec in projects_table['records']:
        pid = rec['values']['project_id']
        req_skill = rec['values']['required_skill']
        projects.append(pid)
        project_required_skill[pid] = req_skill
    offers_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_2_view_0':
            offers_table = t
            break
    if offers_table is None:
        raise RuntimeError('Missing offers table')
    A = []
    cost = {}
    for rec in offers_table['records']:
        wid = rec['values']['worker_id']
        pid = rec['values']['project_id']
        if wid not in eligible_workers:
            continue
        cents = int(rec['values']['cost_cents'])
        w_skill = worker_skill[wid].casefold()
        p_skill = project_required_skill[pid].casefold()
        if skill_order[w_skill] >= skill_order[p_skill]:
            A.append((wid, pid))
            cost[wid, pid] = cents
    for pid in projects:
        found = False
        for (wid, pid2) in A:
            if pid2 == pid:
                found = True
                break
        if not found:
            raise RuntimeError(f'No eligible offer for project {pid}')
    for key in A:
        if key not in cost:
            raise RuntimeError(f'Missing cost for assignment {key}')
    m = gp.Model('AP')
    x = m.addVars(A, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[key] * x[key] for key in A)), GRB.MINIMIZE)
    for pid in projects:
        m.addConstr(gp.quicksum((x[wid, pid] for (wid, pid2) in A if pid2 == pid)) == 1, name='project')
    for wid in eligible_workers:
        m.addConstr(gp.quicksum((x[wid, pid] for (wid2, pid) in A if wid2 == wid)) <= 1, name='worker')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_DATA)