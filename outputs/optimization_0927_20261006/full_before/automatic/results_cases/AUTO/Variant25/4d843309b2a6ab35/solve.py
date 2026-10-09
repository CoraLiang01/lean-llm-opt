import gurobipy as gp
from gurobipy import GRB
workers = [{'worker_id': 'W12', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W06', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W04', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W01', 'skill': 'Junior', 'on_leave': 1}, {'worker_id': 'W11', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W10', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W02', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W00', 'skill': 'Expert', 'on_leave': 0}]
projects = [{'project_id': 'P00', 'required_skill': 'Junior'}, {'project_id': 'P01', 'required_skill': 'Junior'}, {'project_id': 'P02', 'required_skill': 'Junior'}, {'project_id': 'P03', 'required_skill': 'Junior'}, {'project_id': 'P04', 'required_skill': 'Junior'}, {'project_id': 'P05', 'required_skill': 'Intermediate'}]
cost = {'W12': {'P00': 102, 'P01': 353, 'P02': 651, 'P03': 102, 'P04': '', 'P05': 7}, 'W06': {'P00': '', 'P01': 822, 'P02': '', 'P03': 223, 'P04': 1155, 'P05': 1055}, 'W04': {'P00': 642, 'P01': '', 'P02': 1130, 'P03': 133, 'P04': 199, 'P05': 311}, 'W11': {'P00': 1091, 'P01': 324, 'P02': 379, 'P03': 272, 'P04': '', 'P05': ''}, 'W10': {'P00': 1176, 'P01': 1111, 'P02': 1380, 'P03': 542, 'P04': 158, 'P05': 922}, 'W02': {'P00': '', 'P01': 1397, 'P02': 953, 'P03': 714, 'P04': 205, 'P05': ''}, 'W00': {'P00': 1063, 'P01': 219, 'P02': '', 'P03': 1329, 'P04': '', 'P05': 436}}
skill_level = {'Junior': 0, 'Intermediate': 1, 'Senior': 2, 'Expert': 3}
eligible_workers = [w['worker_id'] for w in workers if w['on_leave'] == 0]
project_ids = [p['project_id'] for p in projects]
allowed_pairs = []
cost_pairs = {}
worker_skill = {w['worker_id']: w['skill'] for w in workers}
project_required_skill = {p['project_id']: p['required_skill'] for p in projects}
for w in eligible_workers:
    w_skill = skill_level[worker_skill[w]]
    for p in project_ids:
        p_skill = skill_level[project_required_skill[p]]
        c = cost[w][p]
        if c != '' and w_skill >= p_skill:
            allowed_pairs.append((w, p))
            cost_pairs[w, p] = int(c)
for p in project_ids:
    found = False
    for w in eligible_workers:
        if (w, p) in allowed_pairs:
            found = True
            break
    if not found:
        raise ValueError(f'No eligible worker for project {p}')
for w in eligible_workers:
    pass
m = gp.Model('AP')
x = m.addVars(allowed_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_pairs[w, p] * x[w, p] for (w, p) in allowed_pairs)), GRB.MINIMIZE)
for p in project_ids:
    m.addConstr(gp.quicksum((x[w, p] for w in eligible_workers if (w, p) in allowed_pairs)) == 1, name='prj')
for w in eligible_workers:
    m.addConstr(gp.quicksum((x[w, p] for p in project_ids if (w, p) in allowed_pairs)) <= 1, name='wrk')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')