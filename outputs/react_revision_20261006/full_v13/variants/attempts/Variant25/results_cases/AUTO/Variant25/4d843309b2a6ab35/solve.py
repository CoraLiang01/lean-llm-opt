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

def solve_problem(CSVQA_FRAMES):
    skill_map = {'junior': 1, 'intermediate': 2, 'senior': 3, 'expert': 4}
    workers_frame = CSVQA_FRAMES['file_0_view_0']
    workers = []
    worker_skill = {}
    worker_on_leave = {}
    for (_, row) in workers_frame.iterrows():
        wid = row['worker_id']
        skill = row['skill']
        on_leave = row['on_leave']
        worker_skill[wid] = skill_map[skill.casefold()]
        worker_on_leave[wid] = int(on_leave)
        if int(on_leave) == 0:
            workers.append(wid)
    projects_frame = CSVQA_FRAMES['file_1_view_0']
    projects = []
    project_required_skill = {}
    for (_, row) in projects_frame.iterrows():
        pid = row['project_id']
        req_skill = row['required_skill']
        project_required_skill[pid] = skill_map[req_skill.casefold()]
        projects.append(pid)
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    cost = {}
    eligible_pairs = []
    for (_, row) in cost_frame.iterrows():
        wid = row['worker_id']
        if wid not in workers:
            continue
        for pid in projects:
            cell = row[pid]
            if cell != '' and worker_skill[wid] >= project_required_skill[pid]:
                cost[wid, pid] = int(cell)
                eligible_pairs.append((wid, pid))
    for pid in projects:
        if not any(((wid, pid) in eligible_pairs for wid in workers)):
            raise ValueError(f'No eligible worker for project {pid}')
    m = gp.Model('AP')
    x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[wid, pid] * x_vars[wid, pid] for (wid, pid) in eligible_pairs)), GRB.MINIMIZE)
    for pid in projects:
        m.addConstr(gp.quicksum((x_vars[wid, pid] for wid in workers if (wid, pid) in x_vars)) == 1)
    for wid in workers:
        m.addConstr(gp.quicksum((x_vars[wid, pid] for pid in projects if (wid, pid) in x_vars)) <= 1)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')