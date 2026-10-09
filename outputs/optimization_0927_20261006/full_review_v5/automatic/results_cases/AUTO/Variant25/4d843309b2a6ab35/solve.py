import gurobipy as gp
from gurobipy import GRB
workers = ['W12', 'W06', 'W04', 'W11', 'W10', 'W02', 'W00']
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05']
skill_rank = {'Junior': 0, 'Intermediate': 1, 'Senior': 2, 'Expert': 3}
worker_skill = {'W12': 'Junior', 'W06': 'Expert', 'W04': 'Intermediate', 'W11': 'Junior', 'W10': 'Intermediate', 'W02': 'Intermediate', 'W00': 'Expert'}
project_required_skill = {'P00': 'Junior', 'P01': 'Junior', 'P02': 'Junior', 'P03': 'Junior', 'P04': 'Junior', 'P05': 'Intermediate'}
cost = {'W12': {'P00': 102, 'P01': 353, 'P02': 651, 'P03': 102}, 'W06': {'P01': 822, 'P03': 223, 'P04': 1155, 'P05': 1055}, 'W04': {'P00': 642, 'P02': 1130, 'P03': 133, 'P04': 199, 'P05': 311}, 'W11': {'P00': 1091, 'P01': 324, 'P02': 379, 'P03': 272}, 'W10': {'P00': 1176, 'P01': 1111, 'P02': 1380, 'P03': 542, 'P04': 158, 'P05': 922}, 'W02': {'P01': 1397, 'P02': 953, 'P03': 714, 'P04': 205}, 'W00': {'P00': 1063, 'P01': 219, 'P03': 1329, 'P05': 436}}
eligible_pairs = []
for w in workers:
    for p in projects:
        if p in cost.get(w, {}):
            if skill_rank[worker_skill[w]] >= skill_rank[project_required_skill[p]]:
                eligible_pairs.append((w, p))
for (w, p) in eligible_pairs:
    if w not in cost or p not in cost[w]:
        raise ValueError(f'Missing cost for eligible pair ({w}, {p})')
project_eligible_workers = {p: [w for w in workers if (w, p) in eligible_pairs] for p in projects}
worker_eligible_projects = {w: [p for p in projects if (w, p) in eligible_pairs] for w in workers}
m = gp.Model('assignment')
x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in eligible_pairs)), GRB.MINIMIZE)
for p in projects:
    m.addConstr(gp.quicksum((x_vars[w, p] for w in project_eligible_workers[p])) == 1, name='prj_' + p)
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, p] for p in worker_eligible_projects[w])) <= 1, name='wrk_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')