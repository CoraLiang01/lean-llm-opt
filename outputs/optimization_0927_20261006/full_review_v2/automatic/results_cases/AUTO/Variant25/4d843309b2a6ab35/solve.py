import gurobipy as gp
from gurobipy import GRB
workers = ['W12', 'W06', 'W04', 'W11', 'W10', 'W02', 'W00']
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05']
cost = {'W12': {'P00': 102, 'P01': 353, 'P02': 651, 'P03': 102}, 'W06': {'P01': 822, 'P03': 223, 'P04': 1155, 'P05': 1055}, 'W04': {'P00': 642, 'P02': 1130, 'P03': 133, 'P04': 199, 'P05': 311}, 'W11': {'P00': 1091, 'P01': 324, 'P02': 379, 'P03': 272}, 'W10': {'P00': 1176, 'P01': 1111, 'P02': 1380, 'P03': 542, 'P04': 158, 'P05': 922}, 'W02': {'P01': 1397, 'P02': 953, 'P03': 714, 'P04': 205}, 'W00': {'P00': 1063, 'P01': 219, 'P03': 1329, 'P05': 436}}
allowed_pairs = []
for w in workers:
    for p in projects:
        if w in cost and p in cost[w]:
            allowed_pairs.append((w, p))
for (w, p) in allowed_pairs:
    if not isinstance(cost[w][p], int):
        raise ValueError(f'Cost for ({w},{p}) is not an integer.')
project_to_workers = {p: [] for p in projects}
for (w, p) in allowed_pairs:
    project_to_workers[p].append(w)
worker_to_projects = {w: [] for w in workers}
for (w, p) in allowed_pairs:
    worker_to_projects[w].append(p)
m = gp.Model('Assignment')
x_vars = m.addVars(allowed_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in allowed_pairs)), GRB.MINIMIZE)
for p in projects:
    eligible_ws = project_to_workers[p]
    m.addConstr(gp.quicksum((x_vars[w, p] for w in eligible_ws)) == 1, name='prj_' + p)
for w in workers:
    eligible_ps = worker_to_projects[w]
    m.addConstr(gp.quicksum((x_vars[w, p] for p in eligible_ps)) <= 1, name='wrk_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')