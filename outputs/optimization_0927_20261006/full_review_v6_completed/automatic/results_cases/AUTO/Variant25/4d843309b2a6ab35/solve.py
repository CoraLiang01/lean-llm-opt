import gurobipy as gp
from gurobipy import GRB
managers = [{'worker_id': 'W00', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W02', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W04', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W06', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W10', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W11', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W12', 'skill': 'Junior', 'on_leave': 0}]
projects = [{'project_id': 'P00', 'required_skill': 'Junior'}, {'project_id': 'P01', 'required_skill': 'Junior'}, {'project_id': 'P02', 'required_skill': 'Junior'}, {'project_id': 'P03', 'required_skill': 'Junior'}, {'project_id': 'P04', 'required_skill': 'Junior'}, {'project_id': 'P05', 'required_skill': 'Intermediate'}]
cost = {'W00': {'P00': 1063, 'P01': 219, 'P02': None, 'P03': 1329, 'P04': None, 'P05': 436}, 'W02': {'P00': None, 'P01': 1397, 'P02': 953, 'P03': 714, 'P04': 205, 'P05': None}, 'W04': {'P00': 642, 'P01': None, 'P02': 1130, 'P03': 133, 'P04': 199, 'P05': 311}, 'W06': {'P00': None, 'P01': 822, 'P02': None, 'P03': 223, 'P04': 1155, 'P05': 1055}, 'W10': {'P00': 1176, 'P01': 1111, 'P02': 1380, 'P03': 542, 'P04': 158, 'P05': 922}, 'W11': {'P00': 1091, 'P01': 324, 'P02': 379, 'P03': 272, 'P04': None, 'P05': None}, 'W12': {'P00': 102, 'P01': 353, 'P02': 651, 'P03': 102, 'P04': None, 'P05': 7}}
skill_order = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}
W = [m['worker_id'] for m in managers if m['on_leave'] == 0]
P = [p['project_id'] for p in projects]
worker_skill = {m['worker_id']: skill_order[m['skill']] for m in managers}
project_required_skill = {p['project_id']: skill_order[p['required_skill']] for p in projects}
eligible_pairs = []
for w in W:
    for p in P:
        c = cost[w][p]
        if c is not None and worker_skill[w] >= project_required_skill[p]:
            eligible_pairs.append((w, p))
for p in P:
    if not any(((w, p) in eligible_pairs for w in W)):
        raise ValueError(f'No eligible manager for project {p}')
m = gp.Model('manager_project_assignment')
x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in eligible_pairs)), GRB.MINIMIZE)
for p in P:
    eligible_ws = [w for w in W if (w, p) in eligible_pairs]
    m.addConstr(gp.quicksum((x_vars[w, p] for w in eligible_ws)) == 1, name='prj_' + p)
for w in W:
    eligible_ps = [p for p in P if (w, p) in eligible_pairs]
    m.addConstr(gp.quicksum((x_vars[w, p] for p in eligible_ps)) <= 1, name='mgr_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')