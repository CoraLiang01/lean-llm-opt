CSVQA_DATA = {'query': 'Build a one-job-per-person roster for ten work packages using twelve available candidates. The cost rows are distributed across three files and must be matched using worker_id. Each project needs exactly one person, and each person may take at most one project. Exclude anyone with on_leave=1. The assigned person must meet required_skill, with Junior < Intermediate < Senior < Expert. In the cost matrix, project IDs are columns and worker IDs identify rows. A blank cell forbids that assignment. Minimize the sum of assignment costs, and report the minimum in USD cents.', 'relationships': [], 'tables': [{'columns': ['worker_id', 'skill', 'on_leave'], 'file_name': 'export_01.csv', 'records': [{'data_row_number': 0, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W00'}}, {'data_row_number': 1, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W18'}}, {'data_row_number': 2, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W02'}}, {'data_row_number': 3, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W17'}}, {'data_row_number': 4, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W10'}}, {'data_row_number': 5, 'values': {'on_leave': '0', 'skill': 'Senior', 'worker_id': 'W05'}}, {'data_row_number': 6, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W11'}}, {'data_row_number': 7, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W14'}}, {'data_row_number': 8, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W01'}}, {'data_row_number': 9, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W24'}}, {'data_row_number': 10, 'values': {'on_leave': '0', 'skill': 'Senior', 'worker_id': 'W16'}}], 'table_id': 'file_0_view_0'}, {'columns': ['project_id', 'required_skill'], 'file_name': 'export_02.csv', 'records': [{'data_row_number': 0, 'values': {'project_id': 'P00', 'required_skill': 'Junior'}}, {'data_row_number': 1, 'values': {'project_id': 'P01', 'required_skill': 'Junior'}}, {'data_row_number': 2, 'values': {'project_id': 'P02', 'required_skill': 'Junior'}}, {'data_row_number': 3, 'values': {'project_id': 'P03', 'required_skill': 'Intermediate'}}, {'data_row_number': 4, 'values': {'project_id': 'P04', 'required_skill': 'Junior'}}, {'data_row_number': 5, 'values': {'project_id': 'P05', 'required_skill': 'Junior'}}, {'data_row_number': 6, 'values': {'project_id': 'P06', 'required_skill': 'Junior'}}, {'data_row_number': 7, 'values': {'project_id': 'P07', 'required_skill': 'Expert'}}, {'data_row_number': 8, 'values': {'project_id': 'P08', 'required_skill': 'Junior'}}, {'data_row_number': 9, 'values': {'project_id': 'P09', 'required_skill': 'Intermediate'}}], 'table_id': 'file_1_view_0'}, {'columns': ['worker_id', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09'], 'file_name': 'export_03.csv', 'records': [{'data_row_number': 0, 'values': {'P00': '330', 'P01': '193', 'P02': '', 'P03': '4', 'P04': '431', 'P05': '', 'P06': '1130', 'P07': '15', 'P08': '644', 'P09': '9', 'worker_id': 'W02'}}, {'data_row_number': 1, 'values': {'P00': '', 'P01': '170', 'P02': '', 'P03': '919', 'P04': '660', 'P05': '1314', 'P06': '668', 'P07': '4', 'P08': '462', 'P09': '614', 'worker_id': 'W01'}}, {'data_row_number': 2, 'values': {'P00': '1079', 'P01': '1128', 'P02': '758', 'P03': '', 'P04': '', 'P05': '108', 'P06': '', 'P07': '423', 'P08': '744', 'P09': '347', 'worker_id': 'W10'}}, {'data_row_number': 3, 'values': {'P00': '538', 'P01': '', 'P02': '1359', 'P03': '20', 'P04': '1366', 'P05': '702', 'P06': '122', 'P07': '13', 'P08': '358', 'P09': '', 'worker_id': 'W17'}}], 'table_id': 'file_2_view_0'}, {'columns': ['worker_id', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09'], 'file_name': 'export_04.csv', 'records': [{'data_row_number': 0, 'values': {'P00': '238', 'P01': '', 'P02': '1085', 'P03': '529', 'P04': '582', 'P05': '889', 'P06': '139', 'P07': '18', 'P08': '630', 'P09': '538', 'worker_id': 'W18'}}, {'data_row_number': 1, 'values': {'P00': '325', 'P01': '772', 'P02': '1042', 'P03': '', 'P04': '1394', 'P05': '', 'P06': '374', 'P07': '2', 'P08': '1140', 'P09': '127', 'worker_id': 'W05'}}, {'data_row_number': 2, 'values': {'P00': '', 'P01': '2', 'P02': '4', 'P03': '5', 'P04': '5', 'P05': '14', 'P06': '12', 'P07': '20', 'P08': '17', 'P09': '8', 'worker_id': 'W06'}}, {'data_row_number': 3, 'values': {'P00': '265', 'P01': '1114', 'P02': '', 'P03': '1209', 'P04': '231', 'P05': '221', 'P06': '', 'P07': '17', 'P08': '135', 'P09': '', 'worker_id': 'W16'}}], 'table_id': 'file_3_view_0'}, {'columns': ['worker_id', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09'], 'file_name': 'export_05.csv', 'records': [{'data_row_number': 0, 'values': {'P00': '', 'P01': '1116', 'P02': '414', 'P03': '', 'P04': '554', 'P05': '113', 'P06': '', 'P07': '11', 'P08': '853', 'P09': '1142', 'worker_id': 'W00'}}, {'data_row_number': 1, 'values': {'P00': '1143', 'P01': '841', 'P02': '920', 'P03': '122', 'P04': '634', 'P05': '', 'P06': '1250', 'P07': '836', 'P08': '180', 'P09': '1105', 'worker_id': 'W11'}}, {'data_row_number': 2, 'values': {'P00': '784', 'P01': '1178', 'P02': '1171', 'P03': '', 'P04': '103', 'P05': '137', 'P06': '832', 'P07': '1279', 'P08': '893', 'P09': '351', 'worker_id': 'W24'}}, {'data_row_number': 3, 'values': {'P00': '', 'P01': '687', 'P02': '126', 'P03': '547', 'P04': '125', 'P05': '1358', 'P06': '1391', 'P07': '814', 'P08': '1362', 'P09': '962', 'worker_id': 'W14'}}], 'table_id': 'file_4_view_0'}]}
import gurobipy
from gurobipy import GRB, Model
tables_by_id = {t['table_id']: t for t in CSVQA_DATA['tables']}
skill_map = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}
workers = {}
for rec in tables_by_id['file_0_view_0']['records']:
    wid = rec['values']['worker_id']
    skill_str = rec['values']['skill']
    workers[wid] = {'skill': skill_map[skill_str], 'on_leave': rec['values']['on_leave']}
projects = {}
for rec in tables_by_id['file_1_view_0']['records']:
    pid = rec['values']['project_id']
    req_str = rec['values']['required_skill']
    projects[pid] = {'required_skill': skill_map[req_str]}
costs = {}
for table_id in ['file_2_view_0', 'file_3_view_0', 'file_4_view_0']:
    for rec in tables_by_id[table_id]['records']:
        wid = rec['values']['worker_id']
        for pid in projects:
            val = rec['values'].get(pid, '')
            if val != '':
                costs[wid, pid] = int(val)
allowed = {}
for wid in workers:
    for pid in projects:
        if (wid, pid) in costs and workers[wid]['skill'] >= projects[pid]['required_skill']:
            allowed[wid, pid] = 1
        else:
            allowed[wid, pid] = 0

def build_model():
    m = Model('worker_project_assignment')
    m.setParam(GRB.Param.MIPGap, 1e-06)
    x = {}
    for wid in workers:
        for pid in projects:
            var = m.addVar(vtype=GRB.BINARY, name=f'x_{wid}_{pid}')
            x[wid, pid] = var
    for pid in projects:
        m.addConstr(sum((x[wid, pid] for wid in workers)) == 1, name=f'project_{pid}_assigned_once')
    for wid in workers:
        m.addConstr(sum((x[wid, pid] for pid in projects)) <= 1, name=f'worker_{wid}_max_one')
    for wid in workers:
        for pid in projects:
            m.addConstr(x[wid, pid] <= allowed[wid, pid], name=f'allowed_{wid}_{pid}')
    m.setObjective(sum((costs.get((wid, pid), 0) * x[wid, pid] for wid in workers for pid in projects)), GRB.MINIMIZE)
    m.update()
    m.optimize()
    return m
m = build_model()
