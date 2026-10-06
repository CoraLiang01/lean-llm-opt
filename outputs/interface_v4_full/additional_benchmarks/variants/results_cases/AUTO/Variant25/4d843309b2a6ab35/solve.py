import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    workers = [{'id': 'W12', 'skill': 'Junior', 'on_leave': 0}, {'id': 'W06', 'skill': 'Expert', 'on_leave': 0}, {'id': 'W04', 'skill': 'Intermediate', 'on_leave': 0}, {'id': 'W11', 'skill': 'Junior', 'on_leave': 0}, {'id': 'W10', 'skill': 'Intermediate', 'on_leave': 0}, {'id': 'W02', 'skill': 'Intermediate', 'on_leave': 0}, {'id': 'W00', 'skill': 'Expert', 'on_leave': 0}]
    projects = [{'id': 'P00', 'required_skill': 'Junior'}, {'id': 'P01', 'required_skill': 'Junior'}, {'id': 'P02', 'required_skill': 'Junior'}, {'id': 'P03', 'required_skill': 'Junior'}, {'id': 'P04', 'required_skill': 'Junior'}, {'id': 'P05', 'required_skill': 'Intermediate'}]
    cost = {'W12': {'P00': 102, 'P01': 353, 'P02': 651, 'P03': 102, 'P04': None, 'P05': 7}, 'W06': {'P00': None, 'P01': 822, 'P02': None, 'P03': 223, 'P04': 1155, 'P05': 1055}, 'W04': {'P00': 642, 'P01': None, 'P02': 1130, 'P03': 133, 'P04': 199, 'P05': 311}, 'W11': {'P00': 1091, 'P01': 324, 'P02': 379, 'P03': 272, 'P04': None, 'P05': None}, 'W10': {'P00': 1176, 'P01': 1111, 'P02': 1380, 'P03': 542, 'P04': 158, 'P05': 922}, 'W02': {'P00': None, 'P01': 1397, 'P02': 953, 'P03': 714, 'P04': 205, 'P05': None}, 'W00': {'P00': 1063, 'P01': 219, 'P02': None, 'P03': 1329, 'P04': None, 'P05': 436}}
    skill_order = ['Junior', 'Intermediate', 'Senior', 'Expert']
    skill_level = {s: i for i, s in enumerate(skill_order)}
    worker_ids = [w['id'] for w in workers]
    project_ids = [p['id'] for p in projects]
    worker_skill = {w['id']: w['skill'] for w in workers}
    worker_on_leave = {w['id']: w['on_leave'] for w in workers}
    project_required_skill = {p['id']: p['required_skill'] for p in projects}
    eligible = []
    for i in worker_ids:
        if worker_on_leave[i]:
            continue
        for j in project_ids:
            c = cost[i][j]
            if c is None:
                continue
            if skill_level[worker_skill[i]] < skill_level[project_required_skill[j]]:
                continue
            eligible.append((i, j))
    W_j = {j: [] for j in project_ids}
    for i, j in eligible:
        W_j[j].append(i)
    P_i = {i: [] for i in worker_ids}
    for i, j in eligible:
        P_i[i].append(j)
    for j in project_ids:
        if len(W_j[j]) == 0:
            raise ValueError(f'No eligible worker for project {j}')
    for i, j in eligible:
        if cost[i][j] is None:
            raise ValueError(f'Missing cost for eligible assignment ({i},{j})')
    m = gp.Model('assignment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(eligible, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i, j in eligible)), GRB.MINIMIZE)
    for j in project_ids:
        m.addConstr(gp.quicksum((x[i, j] for i in W_j[j])) == 1, name='prj_' + j)
    for i in worker_ids:
        if len(P_i[i]) > 0:
            m.addConstr(gp.quicksum((x[i, j] for j in P_i[i])) <= 1, name='wrk_' + i)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()