import gurobipy as gp
from gurobipy import GRB
workers = [{'worker_id': 'W02', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W01', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W10', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W17', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W18', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W05', 'skill': 'Senior', 'on_leave': 0}, {'worker_id': 'W16', 'skill': 'Senior', 'on_leave': 0}, {'worker_id': 'W00', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W11', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W24', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W14', 'skill': 'Expert', 'on_leave': 0}]
projects = [{'project_id': 'P00', 'required_skill': 'Junior'}, {'project_id': 'P01', 'required_skill': 'Junior'}, {'project_id': 'P02', 'required_skill': 'Junior'}, {'project_id': 'P03', 'required_skill': 'Intermediate'}, {'project_id': 'P04', 'required_skill': 'Junior'}, {'project_id': 'P05', 'required_skill': 'Junior'}, {'project_id': 'P06', 'required_skill': 'Junior'}, {'project_id': 'P07', 'required_skill': 'Expert'}, {'project_id': 'P08', 'required_skill': 'Junior'}, {'project_id': 'P09', 'required_skill': 'Intermediate'}]
skill_order = ['Junior', 'Intermediate', 'Senior', 'Expert']
cost = {'W02': {'P00': 330, 'P01': 193, 'P03': 4, 'P04': 431, 'P06': 1130, 'P08': 644}, 'W01': {'P01': 170, 'P03': 919, 'P04': 660, 'P05': 1314, 'P06': 668, 'P08': 462, 'P09': 614}, 'W10': {'P00': 1079, 'P01': 1128, 'P02': 758, 'P05': 108, 'P07': 423, 'P08': 744, 'P09': 347}, 'W17': {'P00': 538, 'P02': 1359, 'P04': 1366, 'P05': 702, 'P06': 122, 'P08': 358}, 'W18': {'P00': 238, 'P02': 1085, 'P03': 529, 'P04': 582, 'P05': 889, 'P06': 139, 'P08': 630, 'P09': 538}, 'W05': {'P00': 325, 'P01': 772, 'P02': 1042, 'P04': 1394, 'P06': 374, 'P08': 1140, 'P09': 127}, 'W16': {'P00': 265, 'P01': 1114, 'P03': 1209, 'P04': 231, 'P05': 221, 'P08': 135}, 'W00': {'P01': 1116, 'P02': 414, 'P04': 554, 'P05': 113, 'P08': 853, 'P09': 1142}, 'W11': {'P00': 1143, 'P01': 841, 'P02': 920, 'P03': 122, 'P04': 634, 'P06': 1250, 'P07': 836, 'P08': 180, 'P09': 1105}, 'W24': {'P00': 784, 'P01': 1178, 'P02': 1171, 'P04': 103, 'P05': 137, 'P06': 832, 'P07': 1279, 'P08': 893, 'P09': 351}, 'W14': {'P01': 687, 'P02': 126, 'P03': 547, 'P04': 125, 'P05': 1358, 'P06': 1391, 'P07': 814, 'P08': 1362, 'P09': 962}}
worker_ids = [w['worker_id'] for w in workers]
project_ids = [p['project_id'] for p in projects]
worker_skill = {w['worker_id']: w['skill'] for w in workers}
worker_on_leave = {w['worker_id']: w['on_leave'] for w in workers}
project_required_skill = {p['project_id']: p['required_skill'] for p in projects}
skill_rank = {s: i for (i, s) in enumerate(skill_order)}
eligible_pairs = []
for w in worker_ids:
    if worker_on_leave[w] == 1:
        continue
    for p in project_ids:
        if w in cost and p in cost[w] and (cost[w][p] is not None):
            if skill_rank[worker_skill[w]] >= skill_rank[project_required_skill[p]]:
                eligible_pairs.append((w, p))
for p in project_ids:
    if not any(((w, p) in eligible_pairs for w in worker_ids)):
        raise ValueError(f'No eligible worker for project {p}')
m = gp.Model('worker_project_assignment')
x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in eligible_pairs)), GRB.MINIMIZE)
for p in project_ids:
    m.addConstr(gp.quicksum((x_vars[w, p] for w in worker_ids if (w, p) in x_vars)) == 1, name='prj_' + p)
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, p] for p in project_ids if (w, p) in x_vars)) <= 1, name='wrk_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')