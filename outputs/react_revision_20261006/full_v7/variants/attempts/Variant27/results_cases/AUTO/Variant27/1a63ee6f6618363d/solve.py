CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Build a one-job-per-person roster for ten work packages using twelve available candidates. The cost rows '
          'are distributed across three files and must be matched using worker_id. Each project needs exactly one '
          'person, and each person may take at most one project. Exclude anyone with on_leave=1. The assigned person '
          'must meet required_skill, with Junior < Intermediate < Senior < Expert. In the cost matrix, project IDs are '
          'columns and worker IDs identify rows. A blank cell forbids that assignment. Minimize the sum of assignment '
          'costs, and report the minimum in USD cents.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['worker_id', 'skill', 'on_leave'],
             'file_index': 0,
             'file_name': 'export_01.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W00'}},
                         {'source_row': 1, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W18'}},
                         {'source_row': 2, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W02'}},
                         {'source_row': 3, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W17'}},
                         {'source_row': 4, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W10'}},
                         {'source_row': 5, 'values': {'on_leave': '0', 'skill': 'Senior', 'worker_id': 'W05'}},
                         {'source_row': 6, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W11'}},
                         {'source_row': 7, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W14'}},
                         {'source_row': 8, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W01'}},
                         {'source_row': 9, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W24'}},
                         {'source_row': 10, 'values': {'on_leave': '0', 'skill': 'Senior', 'worker_id': 'W16'}},
                         {'source_row': 11, 'values': {'on_leave': '1', 'skill': 'Junior', 'worker_id': 'W06'}}],
             'returned_rows': 12,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['project_id', 'required_skill'],
             'file_index': 1,
             'file_name': 'export_02.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'project_id': 'P00', 'required_skill': 'Junior'}},
                         {'source_row': 1, 'values': {'project_id': 'P01', 'required_skill': 'Junior'}},
                         {'source_row': 2, 'values': {'project_id': 'P02', 'required_skill': 'Junior'}},
                         {'source_row': 3, 'values': {'project_id': 'P03', 'required_skill': 'Intermediate'}},
                         {'source_row': 4, 'values': {'project_id': 'P04', 'required_skill': 'Junior'}},
                         {'source_row': 5, 'values': {'project_id': 'P05', 'required_skill': 'Junior'}},
                         {'source_row': 6, 'values': {'project_id': 'P06', 'required_skill': 'Junior'}},
                         {'source_row': 7, 'values': {'project_id': 'P07', 'required_skill': 'Expert'}},
                         {'source_row': 8, 'values': {'project_id': 'P08', 'required_skill': 'Junior'}},
                         {'source_row': 9, 'values': {'project_id': 'P09', 'required_skill': 'Intermediate'}}],
             'returned_rows': 10,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['worker_id', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09'],
             'file_index': 2,
             'file_name': 'export_03.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'P00': '330',
                                     'P01': '193',
                                     'P02': '',
                                     'P03': '4',
                                     'P04': '431',
                                     'P05': '',
                                     'P06': '1130',
                                     'P07': '15',
                                     'P08': '644',
                                     'P09': '9',
                                     'worker_id': 'W02'}},
                         {'source_row': 1,
                          'values': {'P00': '',
                                     'P01': '170',
                                     'P02': '',
                                     'P03': '919',
                                     'P04': '660',
                                     'P05': '1314',
                                     'P06': '668',
                                     'P07': '4',
                                     'P08': '462',
                                     'P09': '614',
                                     'worker_id': 'W01'}},
                         {'source_row': 2,
                          'values': {'P00': '1079',
                                     'P01': '1128',
                                     'P02': '758',
                                     'P03': '',
                                     'P04': '',
                                     'P05': '108',
                                     'P06': '',
                                     'P07': '423',
                                     'P08': '744',
                                     'P09': '347',
                                     'worker_id': 'W10'}},
                         {'source_row': 3,
                          'values': {'P00': '538',
                                     'P01': '',
                                     'P02': '1359',
                                     'P03': '20',
                                     'P04': '1366',
                                     'P05': '702',
                                     'P06': '122',
                                     'P07': '13',
                                     'P08': '358',
                                     'P09': '',
                                     'worker_id': 'W17'}}],
             'returned_rows': 4,
             'role': 'file_2',
             'table_id': 'file_2_view_0'},
            {'columns': ['worker_id', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09'],
             'file_index': 3,
             'file_name': 'export_04.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'P00': '238',
                                     'P01': '',
                                     'P02': '1085',
                                     'P03': '529',
                                     'P04': '582',
                                     'P05': '889',
                                     'P06': '139',
                                     'P07': '18',
                                     'P08': '630',
                                     'P09': '538',
                                     'worker_id': 'W18'}},
                         {'source_row': 1,
                          'values': {'P00': '325',
                                     'P01': '772',
                                     'P02': '1042',
                                     'P03': '',
                                     'P04': '1394',
                                     'P05': '',
                                     'P06': '374',
                                     'P07': '2',
                                     'P08': '1140',
                                     'P09': '127',
                                     'worker_id': 'W05'}},
                         {'source_row': 2,
                          'values': {'P00': '',
                                     'P01': '2',
                                     'P02': '4',
                                     'P03': '5',
                                     'P04': '5',
                                     'P05': '14',
                                     'P06': '12',
                                     'P07': '20',
                                     'P08': '17',
                                     'P09': '8',
                                     'worker_id': 'W06'}},
                         {'source_row': 3,
                          'values': {'P00': '265',
                                     'P01': '1114',
                                     'P02': '',
                                     'P03': '1209',
                                     'P04': '231',
                                     'P05': '221',
                                     'P06': '',
                                     'P07': '17',
                                     'P08': '135',
                                     'P09': '',
                                     'worker_id': 'W16'}}],
             'returned_rows': 4,
             'role': 'file_3',
             'table_id': 'file_3_view_0'},
            {'columns': ['worker_id', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09'],
             'file_index': 4,
             'file_name': 'export_05.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'P00': '',
                                     'P01': '1116',
                                     'P02': '414',
                                     'P03': '',
                                     'P04': '554',
                                     'P05': '113',
                                     'P06': '',
                                     'P07': '11',
                                     'P08': '853',
                                     'P09': '1142',
                                     'worker_id': 'W00'}},
                         {'source_row': 1,
                          'values': {'P00': '1143',
                                     'P01': '841',
                                     'P02': '920',
                                     'P03': '122',
                                     'P04': '634',
                                     'P05': '',
                                     'P06': '1250',
                                     'P07': '836',
                                     'P08': '180',
                                     'P09': '1105',
                                     'worker_id': 'W11'}},
                         {'source_row': 2,
                          'values': {'P00': '784',
                                     'P01': '1178',
                                     'P02': '1171',
                                     'P03': '',
                                     'P04': '103',
                                     'P05': '137',
                                     'P06': '832',
                                     'P07': '1279',
                                     'P08': '893',
                                     'P09': '351',
                                     'worker_id': 'W24'}},
                         {'source_row': 3,
                          'values': {'P00': '',
                                     'P01': '687',
                                     'P02': '126',
                                     'P03': '547',
                                     'P04': '125',
                                     'P05': '1358',
                                     'P06': '1391',
                                     'P07': '814',
                                     'P08': '1362',
                                     'P09': '962',
                                     'worker_id': 'W14'}}],
             'returned_rows': 4,
             'role': 'file_4',
             'table_id': 'file_4_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [4, 10], "
                                   "'expected_shape': [11, 10], 'row_ids_aligned': False, 'column_ids_aligned': True, "
                                   "'row_mapping_basis': 'unresolved', 'column_mapping_basis': 'exact'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [4, 10], "
                                   "'expected_shape': [11, 10], 'row_ids_aligned': False, 'column_ids_aligned': True, "
                                   "'row_mapping_basis': 'unresolved', 'column_mapping_basis': 'exact'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    skill_order = {'junior': 0, 'intermediate': 1, 'senior': 2, 'expert': 3}
    df_workers = CSVQA_FRAMES['file_0_view_0']
    all_workers = df_workers['worker_id'].tolist()
    workers_on_leave = df_workers[df_workers['on_leave'].str.strip() == '1']['worker_id'].tolist()
    workers = df_workers[df_workers['on_leave'].str.strip() == '0']['worker_id'].tolist()
    worker_skill = {row['worker_id']: skill_order[row['skill'].strip().casefold()] for (_, row) in df_workers.iterrows()}
    df_projects = CSVQA_FRAMES['file_1_view_0']
    projects = df_projects['project_id'].tolist()
    project_required_skill = {row['project_id']: skill_order[row['required_skill'].strip().casefold()] for (_, row) in df_projects.iterrows()}
    cost_tables = ['file_2_view_0', 'file_3_view_0', 'file_4_view_0']
    cost_entries = []
    for table_id in cost_tables:
        df = CSVQA_FRAMES[table_id]
        for (_, row) in df.iterrows():
            w = row['worker_id']
            for p in projects:
                val = row[p]
                if val is not None and str(val).strip() != '':
                    cost_entries.append((w, p, int(str(val).strip())))
    cost_dict = {}
    for (w, p, c) in cost_entries:
        cost_dict[w, p] = c
    feasible_wp = []
    c_wp = {}
    for w in workers:
        for p in projects:
            if (w, p) in cost_dict:
                if worker_skill[w] >= project_required_skill[p]:
                    feasible_wp.append((w, p))
                    c_wp[w, p] = cost_dict[w, p]
    for p in projects:
        if not any(((w, p) in feasible_wp for w in workers)):
            raise ValueError(f'No feasible worker for project {p}')
    m = gp.Model('AP')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(feasible_wp, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_wp[w, p] * x_vars[w, p] for (w, p) in feasible_wp)), GRB.MINIMIZE)
    for p in projects:
        m.addConstr(gp.quicksum((x_vars[w, p] for w in workers if (w, p) in feasible_wp)) == 1, name='proj')
    for w in workers:
        m.addConstr(gp.quicksum((x_vars[w, p] for p in projects if (w, p) in feasible_wp)) <= 1, name='work')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')