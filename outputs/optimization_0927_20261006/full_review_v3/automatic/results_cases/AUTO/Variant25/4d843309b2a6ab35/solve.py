import gurobipy as gp
from gurobipy import GRB
workers = [{'worker_id': 'W12', 'skill': 'Junior', 'on_leave': '0'}, {'worker_id': 'W06', 'skill': 'Expert', 'on_leave': '0'}, {'worker_id': 'W04', 'skill': 'Intermediate', 'on_leave': '0'}, {'worker_id': 'W01', 'skill': 'Junior', 'on_leave': '1'}, {'worker_id': 'W11', 'skill': 'Junior', 'on_leave': '0'}, {'worker_id': 'W10', 'skill': 'Intermediate', 'on_leave': '0'}, {'worker_id': 'W02', 'skill': 'Intermediate', 'on_leave': '0'}, {'worker_id': 'W00', 'skill': 'Expert', 'on_leave': '0'}]
projects = [{'project_id': 'P00', 'required_skill': 'Junior'}, {'project_id': 'P01', 'required_skill': 'Junior'}, {'project_id': 'P02', 'required_skill': 'Junior'}, {'project_id': 'P03', 'required_skill': 'Junior'}, {'project_id': 'P04', 'required_skill': 'Junior'}, {'project_id': 'P05', 'required_skill': 'Intermediate'}]
cost_matrix = {'W11': {'P00': '1091', 'P01': '324', 'P02': '379', 'P03': '272', 'P04': '', 'P05': ''}, 'W10': {'P00': '1176', 'P01': '1111', 'P02': '1380', 'P03': '542', 'P04': '158', 'P05': '922'}, 'W06': {'P00': '', 'P01': '822', 'P02': '', 'P03': '223', 'P04': '1155', 'P05': '1055'}, 'W01': {'P00': '5', 'P01': '', 'P02': '20', 'P03': '', 'P04': '18', 'P05': '7'}, 'W02': {'P00': '', 'P01': '1397', 'P02': '953', 'P03': '714', 'P04': '205', 'P05': ''}, 'W00': {'P00': '1063', 'P01': '219', 'P02': '', 'P03': '1329', 'P04': '', 'P05': '436'}, 'W04': {'P00': '642', 'P01': '', 'P02': '1130', 'P03': '133', 'P04': '199', 'P05': '311'}, 'W12': {'P00': '102', 'P01': '353', 'P02': '651', 'P03': '102', 'P04': '', 'P05': '7'}}
skill_level = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}
worker_ids = [w['worker_id'] for w in workers if w['on_leave'] == '0']
project_ids = [p['project_id'] for p in projects]
eligible_pairs = []
cost = {}
for w in workers:
    wid = w['worker_id']
    if w['on_leave'] != '0':
        continue
    w_skill = skill_level[w['skill']]
    for p in projects:
        pid = p['project_id']
        p_skill = skill_level[p['required_skill']]
        cstr = cost_matrix[wid][pid]
        if cstr == '':
            continue
        if w_skill < p_skill:
            continue
        eligible_pairs.append((wid, pid))
        if wid not in cost:
            cost[wid] = {}
        cost[wid][pid] = int(cstr)
for wid in worker_ids:
    for pid in project_ids:
        if (wid, pid) in eligible_pairs:
            if wid not in cost or pid not in cost[wid]:
                raise ValueError(f'Missing cost for eligible pair ({wid},{pid})')
W_p = {}
for p in project_ids:
    W_p[p] = [w for w in worker_ids if (w, p) in eligible_pairs]
    if len(W_p[p]) == 0:
        raise ValueError(f'No eligible workers for project {p}')
P_w = {}
for w in worker_ids:
    P_w[w] = [p for p in project_ids if (w, p) in eligible_pairs]
m = gp.Model('project_assignment')
x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in eligible_pairs)), GRB.MINIMIZE)
for p in project_ids:
    m.addConstr(gp.quicksum((x_vars[w, p] for w in W_p[p])) == 1, name='prj_' + p)
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, p] for p in P_w[w])) <= 1, name='wrk_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {int(m.ObjVal)}')
    for v in m.getVars():
        print(f'{v.VarName}: {int(v.X)}')
else:
    print(f'Solver status: {m.Status}')