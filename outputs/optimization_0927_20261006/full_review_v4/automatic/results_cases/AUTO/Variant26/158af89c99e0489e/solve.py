import gurobipy as gp
from gurobipy import GRB
offers = [{'worker_id': 'W10', 'project_id': 'P00', 'cost_cents': 147, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W10', 'project_id': 'P01', 'cost_cents': 114, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W10', 'project_id': 'P03', 'cost_cents': 998, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W10', 'project_id': 'P04', 'cost_cents': 1270, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W10', 'project_id': 'P07', 'cost_cents': 953, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P03', 'cost_cents': 109, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P04', 'cost_cents': 1306, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P05', 'cost_cents': 626, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P06', 'cost_cents': 466, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P07', 'cost_cents': 1365, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P00', 'cost_cents': 235, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Intermediate'}, {'worker_id': 'W11', 'project_id': 'P02', 'cost_cents': 1034, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Senior'}, {'worker_id': 'W11', 'project_id': 'P03', 'cost_cents': 782, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P04', 'cost_cents': 556, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P05', 'cost_cents': 1018, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P07', 'cost_cents': 1136, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W00', 'project_id': 'P04', 'cost_cents': 430, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W00', 'project_id': 'P05', 'cost_cents': 1385, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W00', 'project_id': 'P06', 'cost_cents': 260, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W00', 'project_id': 'P07', 'cost_cents': 1107, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P00', 'cost_cents': 675, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Intermediate'}, {'worker_id': 'W20', 'project_id': 'P01', 'cost_cents': 1055, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Intermediate'}, {'worker_id': 'W20', 'project_id': 'P03', 'cost_cents': 703, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P06', 'cost_cents': 893, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P07', 'cost_cents': 129, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P03', 'cost_cents': 543, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P04', 'cost_cents': 205, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P05', 'cost_cents': 1149, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P06', 'cost_cents': 533, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P07', 'cost_cents': 732, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P00', 'cost_cents': 1217, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W06', 'project_id': 'P01', 'cost_cents': 425, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W06', 'project_id': 'P03', 'cost_cents': 1097, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P04', 'cost_cents': 822, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P05', 'cost_cents': 221, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P06', 'cost_cents': 496, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P07', 'cost_cents': 908, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P03', 'cost_cents': 379, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P04', 'cost_cents': 239, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P05', 'cost_cents': 460, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P06', 'cost_cents': 130, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P00', 'cost_cents': 722, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W15', 'project_id': 'P03', 'cost_cents': 1062, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P04', 'cost_cents': 1314, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P05', 'cost_cents': 528, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P06', 'cost_cents': 835, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P07', 'cost_cents': 918, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}]
skill_rank = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}
E = []
cost = {}
workers = set()
projects = set()
for offer in offers:
    w = offer['worker_id']
    p = offer['project_id']
    c = offer['cost_cents']
    l = offer['on_leave']
    s_w = offer['worker_skill']
    r_p = offer['required_skill']
    workers.add(w)
    projects.add(p)
    if l == 0 and skill_rank[s_w] >= skill_rank[r_p]:
        E.append((w, p))
        cost[w, p] = c
workers = sorted(workers)
projects = sorted(projects)
for p in projects:
    if not any(((w, p) in E for w in workers)):
        raise ValueError(f'No eligible worker for project {p}')
m = gp.Model('assignment')
x_vars = m.addVars(E, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, p] * x_vars[w, p] for (w, p) in E)), GRB.MINIMIZE)
for p in projects:
    m.addConstr(gp.quicksum((x_vars[w, p] for w in workers if (w, p) in E)) == 1, name='prj_' + p)
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, p] for p in projects if (w, p) in E)) <= 1, name='wrk_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')