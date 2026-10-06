import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    workers = [{'worker_id': 'W12', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W06', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W04', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W11', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W10', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W02', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W00', 'skill': 'Expert', 'on_leave': 0}]
    projects = [{'project_id': 'P00', 'required_skill': 'Junior'}, {'project_id': 'P01', 'required_skill': 'Junior'}, {'project_id': 'P02', 'required_skill': 'Junior'}, {'project_id': 'P03', 'required_skill': 'Junior'}, {'project_id': 'P04', 'required_skill': 'Junior'}, {'project_id': 'P05', 'required_skill': 'Intermediate'}]
    cost = {'W12': {'P00': 102, 'P01': 353, 'P02': 651, 'P03': 102, 'P05': 7}, 'W06': {'P01': 822, 'P03': 223, 'P04': 1155, 'P05': 1055}, 'W04': {'P00': 642, 'P02': 1130, 'P03': 133, 'P04': 199, 'P05': 311}, 'W11': {'P00': 1091, 'P01': 324, 'P02': 379, 'P03': 272}, 'W10': {'P00': 1176, 'P01': 1111, 'P02': 1380, 'P03': 542, 'P04': 158, 'P05': 922}, 'W02': {'P01': 1397, 'P02': 953, 'P03': 714, 'P04': 205}, 'W00': {'P00': 1063, 'P01': 219, 'P03': 1329, 'P05': 436}}
    skill_order = ['Junior', 'Intermediate', 'Senior', 'Expert']
    worker_ids = [w['worker_id'] for w in workers]
    project_ids = [p['project_id'] for p in projects]
    worker_skill = {w['worker_id']: w['skill'] for w in workers}
    project_required_skill = {p['project_id']: p['required_skill'] for p in projects}
    eligible_pairs = []
    for w in worker_ids:
        for p in project_ids:
            if p in cost.get(w, {}):
                if skill_order.index(worker_skill[w]) >= skill_order.index(project_required_skill[p]):
                    eligible_pairs.append((w, p))
    W_p = {p: [] for p in project_ids}
    for w, p in eligible_pairs:
        W_p[p].append(w)
    P_w = {w: [] for w in worker_ids}
    for w, p in eligible_pairs:
        P_w[w].append(p)
    for p in project_ids:
        if len(W_p[p]) == 0:
            raise ValueError(f'No eligible workers for project {p}')
    m = gp.Model('assignment')
    x = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w][p] * x[w, p] for w, p in eligible_pairs)), GRB.MINIMIZE)
    for p in project_ids:
        m.addConstr(gp.quicksum((x[w, p] for w in W_p[p])) == 1, name=f'prj_{p}')
    for w in worker_ids:
        if len(P_w[w]) > 0:
            m.addConstr(gp.quicksum((x[w, p] for p in P_w[w])) <= 1, name=f'wrk_{w}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()