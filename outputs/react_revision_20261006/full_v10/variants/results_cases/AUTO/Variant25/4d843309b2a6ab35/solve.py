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
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    skill_order = {'junior': 0, 'intermediate': 1, 'senior': 2, 'expert': 3}
    workers_frame = CSVQA_FRAMES['file_0_view_0']
    workers = []
    workers_on_leave = []
    worker_skill = {}
    for (_, row) in workers_frame.iterrows():
        wid = row['worker_id']
        skl = row['skill']
        on_leave = row['on_leave']
        if on_leave == '1':
            workers_on_leave.append(wid)
        else:
            workers.append(wid)
        worker_skill[wid] = skl
    projects_frame = CSVQA_FRAMES['file_1_view_0']
    projects = []
    project_required_skill = {}
    for (_, row) in projects_frame.iterrows():
        pid = row['project_id']
        rskl = row['required_skill']
        projects.append(pid)
        project_required_skill[pid] = rskl
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    cost = {}
    for (_, row) in cost_frame.iterrows():
        wid = row['worker_id']
        for pid in projects:
            val = row[pid]
            if val != '':
                cost[wid, pid] = int(val)
    eligible_pairs = []
    for wid in workers:
        w_skill = worker_skill[wid]
        w_skill_ord = skill_order[w_skill.casefold()]
        for pid in projects:
            p_req_skill = project_required_skill[pid]
            p_req_skill_ord = skill_order[p_req_skill.casefold()]
            if (wid, pid) in cost:
                if w_skill_ord >= p_req_skill_ord:
                    eligible_pairs.append((wid, pid))
    m = gp.Model('assignment')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[wid, pid] * x_vars[wid, pid] for (wid, pid) in eligible_pairs)), GRB.MINIMIZE)
    for pid in projects:
        eligible_wids = [wid for wid in workers if (wid, pid) in x_vars]
        m.addConstr(gp.quicksum((x_vars[wid, pid] for wid in eligible_wids)) == 1)
    for wid in workers:
        eligible_pids = [pid for pid in projects if (wid, pid) in x_vars]
        m.addConstr(gp.quicksum((x_vars[wid, pid] for pid in eligible_pids)) <= 1)
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')