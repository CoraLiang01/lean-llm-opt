import gurobipy as gp
from gurobipy import GRB
workers = [{'worker_id': 'W00', 'skill': 'Intermediate', 'on_leave': '0'}, {'worker_id': 'W18', 'skill': 'Intermediate', 'on_leave': '0'}, {'worker_id': 'W02', 'skill': 'Junior', 'on_leave': '0'}, {'worker_id': 'W17', 'skill': 'Junior', 'on_leave': '0'}, {'worker_id': 'W10', 'skill': 'Expert', 'on_leave': '0'}, {'worker_id': 'W05', 'skill': 'Senior', 'on_leave': '0'}, {'worker_id': 'W11', 'skill': 'Expert', 'on_leave': '0'}, {'worker_id': 'W14', 'skill': 'Expert', 'on_leave': '0'}, {'worker_id': 'W01', 'skill': 'Intermediate', 'on_leave': '0'}, {'worker_id': 'W24', 'skill': 'Expert', 'on_leave': '0'}, {'worker_id': 'W16', 'skill': 'Senior', 'on_leave': '0'}]
projects = [{'project_id': 'P00', 'required_skill': 'Junior'}, {'project_id': 'P01', 'required_skill': 'Junior'}, {'project_id': 'P02', 'required_skill': 'Junior'}, {'project_id': 'P03', 'required_skill': 'Intermediate'}, {'project_id': 'P04', 'required_skill': 'Junior'}, {'project_id': 'P05', 'required_skill': 'Junior'}, {'project_id': 'P06', 'required_skill': 'Junior'}, {'project_id': 'P07', 'required_skill': 'Expert'}, {'project_id': 'P08', 'required_skill': 'Junior'}, {'project_id': 'P09', 'required_skill': 'Intermediate'}]
cost = {'W00': {'P00': '', 'P01': '1116', 'P02': '414', 'P03': '', 'P04': '554', 'P05': '113', 'P06': '', 'P07': '11', 'P08': '853', 'P09': '1142'}, 'W01': {'P00': '', 'P01': '170', 'P02': '', 'P03': '919', 'P04': '660', 'P05': '1314', 'P06': '668', 'P07': '4', 'P08': '462', 'P09': '614'}, 'W02': {'P00': '330', 'P01': '193', 'P02': '', 'P03': '4', 'P04': '431', 'P05': '', 'P06': '1130', 'P07': '15', 'P08': '644', 'P09': '9'}, 'W05': {'P00': '325', 'P01': '772', 'P02': '1042', 'P03': '', 'P04': '1394', 'P05': '', 'P06': '374', 'P07': '2', 'P08': '1140', 'P09': '127'}, 'W10': {'P00': '1079', 'P01': '1128', 'P02': '758', 'P03': '', 'P04': '', 'P05': '108', 'P06': '', 'P07': '423', 'P08': '744', 'P09': '347'}, 'W11': {'P00': '1143', 'P01': '841', 'P02': '920', 'P03': '122', 'P04': '634', 'P05': '', 'P06': '1250', 'P07': '836', 'P08': '180', 'P09': '1105'}, 'W14': {'P00': '', 'P01': '687', 'P02': '126', 'P03': '547', 'P04': '125', 'P05': '1358', 'P06': '1391', 'P07': '814', 'P08': '1362', 'P09': '962'}, 'W16': {'P00': '265', 'P01': '1114', 'P02': '', 'P03': '1209', 'P04': '231', 'P05': '221', 'P06': '', 'P07': '17', 'P08': '135', 'P09': ''}, 'W17': {'P00': '538', 'P01': '', 'P02': '1359', 'P03': '20', 'P04': '1366', 'P05': '702', 'P06': '122', 'P07': '13', 'P08': '358', 'P09': ''}, 'W18': {'P00': '238', 'P01': '', 'P02': '1085', 'P03': '529', 'P04': '582', 'P05': '889', 'P06': '139', 'P07': '18', 'P08': '630', 'P09': '538'}, 'W24': {'P00': '784', 'P01': '1178', 'P02': '1171', 'P03': '', 'P04': '103', 'P05': '137', 'P06': '832', 'P07': '1279', 'P08': '893', 'P09': '351'}}
skill_order = ['Junior', 'Intermediate', 'Senior', 'Expert']
worker_ids = [w['worker_id'] for w in workers if w['on_leave'] == '0']
project_ids = [p['project_id'] for p in projects]
skill_level = {s: i for (i, s) in enumerate(skill_order)}
eligible_pairs = []
cost_matrix = {}
for w in workers:
    wid = w['worker_id']
    if w['on_leave'] != '0':
        continue
    w_skill = w['skill']
    w_level = skill_level[w_skill]
    cost_matrix[wid] = {}
    for p in projects:
        pid = p['project_id']
        p_skill = p['required_skill']
        p_level = skill_level[p_skill]
        cstr = cost[wid][pid]
        if cstr == '':
            continue
        if w_level < p_level:
            continue
        eligible_pairs.append((wid, pid))
        cost_matrix[wid][pid] = int(cstr)
for (wid, pid) in eligible_pairs:
    if wid not in cost_matrix or pid not in cost_matrix[wid]:
        raise ValueError(f'Missing cost for eligible pair ({wid},{pid})')
m = gp.Model('worker_project_assignment')
x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_matrix[wid][pid] * x_vars[wid, pid] for (wid, pid) in eligible_pairs)), GRB.MINIMIZE)
for pid in project_ids:
    eligible_ws = [wid for (wid, pid2) in eligible_pairs if pid2 == pid]
    m.addConstr(gp.quicksum((x_vars[wid, pid] for wid in eligible_ws)) == 1, name='prj_' + pid)
for wid in worker_ids:
    eligible_ps = [pid for (wid2, pid) in eligible_pairs if wid2 == wid]
    m.addConstr(gp.quicksum((x_vars[wid, pid] for pid in eligible_ps)) <= 1, name='wrk_' + wid)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')