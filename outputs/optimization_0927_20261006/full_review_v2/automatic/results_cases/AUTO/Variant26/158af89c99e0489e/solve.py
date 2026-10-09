import gurobipy as gp
from gurobipy import GRB
skill_levels = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}
offers_data = [{'worker_id': 'W10', 'project_id': 'P00', 'cost_cents': 147, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W10', 'project_id': 'P01', 'cost_cents': 114, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W10', 'project_id': 'P03', 'cost_cents': 998, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W10', 'project_id': 'P04', 'cost_cents': 1270, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W10', 'project_id': 'P07', 'cost_cents': 953, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P00', 'cost_cents': 5, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Intermediate'}, {'worker_id': 'W19', 'project_id': 'P02', 'cost_cents': 9, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Senior'}, {'worker_id': 'W19', 'project_id': 'P03', 'cost_cents': 109, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P04', 'cost_cents': 1306, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P05', 'cost_cents': 626, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P06', 'cost_cents': 466, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W19', 'project_id': 'P07', 'cost_cents': 1365, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P00', 'cost_cents': 235, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Intermediate'}, {'worker_id': 'W11', 'project_id': 'P02', 'cost_cents': 1034, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Senior'}, {'worker_id': 'W11', 'project_id': 'P03', 'cost_cents': 782, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P04', 'cost_cents': 556, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P05', 'cost_cents': 1018, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P07', 'cost_cents': 1136, 'on_leave': 0, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W00', 'project_id': 'P01', 'cost_cents': 17, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Intermediate'}, {'worker_id': 'W00', 'project_id': 'P04', 'cost_cents': 430, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W00', 'project_id': 'P05', 'cost_cents': 1385, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W00', 'project_id': 'P06', 'cost_cents': 260, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W00', 'project_id': 'P07', 'cost_cents': 1107, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P00', 'cost_cents': 675, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Intermediate'}, {'worker_id': 'W20', 'project_id': 'P01', 'cost_cents': 1055, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Intermediate'}, {'worker_id': 'W20', 'project_id': 'P02', 'cost_cents': 8, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Senior'}, {'worker_id': 'W20', 'project_id': 'P03', 'cost_cents': 703, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P06', 'cost_cents': 893, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P07', 'cost_cents': 129, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P00', 'cost_cents': 4, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Intermediate'}, {'worker_id': 'W04', 'project_id': 'P01', 'cost_cents': 8, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Intermediate'}, {'worker_id': 'W04', 'project_id': 'P02', 'cost_cents': 16, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Senior'}, {'worker_id': 'W04', 'project_id': 'P03', 'cost_cents': 543, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P04', 'cost_cents': 205, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P05', 'cost_cents': 1149, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P06', 'cost_cents': 533, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W04', 'project_id': 'P07', 'cost_cents': 732, 'on_leave': 0, 'worker_skill': 'Junior', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P00', 'cost_cents': 1217, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W06', 'project_id': 'P01', 'cost_cents': 425, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W06', 'project_id': 'P02', 'cost_cents': 1320, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Senior'}, {'worker_id': 'W06', 'project_id': 'P03', 'cost_cents': 1097, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P04', 'cost_cents': 822, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P05', 'cost_cents': 221, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P06', 'cost_cents': 496, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P07', 'cost_cents': 908, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P01', 'cost_cents': 1361, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Intermediate'}, {'worker_id': 'W14', 'project_id': 'P03', 'cost_cents': 379, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P04', 'cost_cents': 239, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P05', 'cost_cents': 460, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P06', 'cost_cents': 130, 'on_leave': 0, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P00', 'cost_cents': 722, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W15', 'project_id': 'P02', 'cost_cents': 577, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Senior'}, {'worker_id': 'W15', 'project_id': 'P03', 'cost_cents': 1062, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P04', 'cost_cents': 1314, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P05', 'cost_cents': 528, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P06', 'cost_cents': 835, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P07', 'cost_cents': 918, 'on_leave': 0, 'worker_skill': 'Expert', 'required_skill': 'Junior'}]
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
workers = ['W00', 'W04', 'W06', 'W10', 'W11', 'W14', 'W15', 'W19', 'W20']
O = []
cost = {}
S_w = {}
R_p = {}
for offer in offers_data:
    w = offer['worker_id']
    p = offer['project_id']
    if offer['on_leave'] != 0:
        continue
    O.append((w, p))
    cost[w, p] = offer['cost_cents']
    S_w[w] = skill_levels[offer['worker_skill']]
    R_p[p] = skill_levels[offer['required_skill']]
for (w, p) in O:
    if w not in workers:
        raise ValueError(f'Worker {w} in offers but not in workers list')
    if p not in projects:
        raise ValueError(f'Project {p} in offers but not in projects list')
project_eligible = {p: [] for p in projects}
worker_eligible = {w: [] for w in workers}
for (w, p) in O:
    if S_w[w] >= R_p[p]:
        project_eligible[p].append((w, p))
        worker_eligible[w].append((w, p))
m = gp.Model('assignment')
x_vars = m.addVars(O, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, p] * x_vars[w, p] for (w, p) in O)), GRB.MINIMIZE)
for p in projects:
    eligible = [pair for pair in project_eligible[p]]
    m.addConstr(gp.quicksum((x_vars[w, p] for (w, p) in eligible)) == 1, name='cov_' + p)
for w in workers:
    eligible = [pair for pair in worker_eligible[w]]
    m.addConstr(gp.quicksum((x_vars[w, p] for (w, p) in eligible)) <= 1, name='asg_' + w)
for (w, p) in O:
    if S_w[w] < R_p[p]:
        m.addConstr(x_vars[w, p] == 0, name='skill_' + w + '_' + p)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')