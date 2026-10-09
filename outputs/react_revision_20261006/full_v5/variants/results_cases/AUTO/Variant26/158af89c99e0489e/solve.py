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
             'filters': {'conditions': [{'column': 'on_leave',
                                         'dtype': 'string',
                                         'evidence': 'Exclude anyone with on_leave=1.',
                                         'inclusive': 'both',
                                         'operator': 'eq',
                                         'value': '0'}],
                         'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 1, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W00'}},
                         {'source_row': 2, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W19'}},
                         {'source_row': 3, 'values': {'on_leave': '0', 'skill': 'Senior', 'worker_id': 'W11'}},
                         {'source_row': 4, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W04'}},
                         {'source_row': 5, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W15'}},
                         {'source_row': 6, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W10'}},
                         {'source_row': 7, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W06'}},
                         {'source_row': 8, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W20'}},
                         {'source_row': 9, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W14'}}],
             'returned_rows': 9,
             'role': 'worker attributes',
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
             'role': 'project requirements',
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
             'role': 'offer list',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    skill_order = ['Junior', 'Intermediate', 'Senior', 'Expert']
    skill_level = {s: i + 1 for (i, s) in enumerate(skill_order)}
    workers_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            workers_table = t
            break
    if workers_table is None:
        raise RuntimeError('Missing workers table')
    workers = []
    worker_skill = {}
    for rec in workers_table['records']:
        w = rec['values']['worker_id']
        s = rec['values']['skill']
        workers.append(w)
        worker_skill[w] = skill_level[s.casefold().capitalize() if s.casefold() in [x.casefold() for x in skill_order] else s]
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
        p = rec['values']['project_id']
        s = rec['values']['required_skill']
        projects.append(p)
        project_required_skill[p] = skill_level[s.casefold().capitalize() if s.casefold() in [x.casefold() for x in skill_order] else s]
    offers_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_2_view_0':
            offers_table = t
            break
    if offers_table is None:
        raise RuntimeError('Missing offers table')
    offer_keys = []
    offer_cost = {}
    for rec in offers_table['records']:
        w = rec['values']['worker_id']
        p = rec['values']['project_id']
        c = int(rec['values']['cost_cents'])
        if w in workers and p in projects:
            if worker_skill[w] >= project_required_skill[p]:
                offer_keys.append((w, p))
                offer_cost[w, p] = c
    for p in projects:
        found = False
        for (w, pp) in offer_keys:
            if pp == p:
                found = True
                break
        if not found:
            raise RuntimeError(f'No eligible offer for project {p}')
    for k in offer_keys:
        if k not in offer_cost:
            raise RuntimeError(f'Missing cost for {k}')
    m = gp.Model('Original_RAG_AP')
    x = m.addVars(offer_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((offer_cost[w_p] * x[w_p] for w_p in offer_keys)), GRB.MINIMIZE)
    for p in projects:
        m.addConstr(gp.quicksum((x[w, p] for (w, pp) in offer_keys if pp == p)) == 1, name='prj')
    for w in workers:
        m.addConstr(gp.quicksum((x[w, p] for (ww, p) in offer_keys if ww == w)) <= 1, name='wrk')
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