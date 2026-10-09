import gurobipy as gp
from gurobipy import GRB
offers = [{'worker_id': 'W10', 'project_id': 'P00', 'cost_cents': 147, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W10', 'project_id': 'P01', 'cost_cents': 114, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W10', 'project_id': 'P03', 'cost_cents': 998, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W10', 'project_id': 'P04', 'cost_cents': 1270, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W10', 'project_id': 'P07', 'cost_cents': 953, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P00', 'cost_cents': 235, 'worker_skill': 'Senior', 'required_skill': 'Intermediate'}, {'worker_id': 'W11', 'project_id': 'P02', 'cost_cents': 1034, 'worker_skill': 'Senior', 'required_skill': 'Senior'}, {'worker_id': 'W11', 'project_id': 'P03', 'cost_cents': 782, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P04', 'cost_cents': 556, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P05', 'cost_cents': 1018, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W11', 'project_id': 'P07', 'cost_cents': 1136, 'worker_skill': 'Senior', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P00', 'cost_cents': 675, 'worker_skill': 'Intermediate', 'required_skill': 'Intermediate'}, {'worker_id': 'W20', 'project_id': 'P01', 'cost_cents': 1055, 'worker_skill': 'Intermediate', 'required_skill': 'Intermediate'}, {'worker_id': 'W20', 'project_id': 'P02', 'cost_cents': 8, 'worker_skill': 'Intermediate', 'required_skill': 'Senior'}, {'worker_id': 'W20', 'project_id': 'P03', 'cost_cents': 703, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P06', 'cost_cents': 893, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W20', 'project_id': 'P07', 'cost_cents': 129, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P01', 'cost_cents': 1361, 'worker_skill': 'Intermediate', 'required_skill': 'Intermediate'}, {'worker_id': 'W14', 'project_id': 'P03', 'cost_cents': 379, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P04', 'cost_cents': 239, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P05', 'cost_cents': 460, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W14', 'project_id': 'P06', 'cost_cents': 130, 'worker_skill': 'Intermediate', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P00', 'cost_cents': 722, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W15', 'project_id': 'P02', 'cost_cents': 577, 'worker_skill': 'Expert', 'required_skill': 'Senior'}, {'worker_id': 'W15', 'project_id': 'P03', 'cost_cents': 1062, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P04', 'cost_cents': 1314, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P05', 'cost_cents': 528, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P06', 'cost_cents': 835, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W15', 'project_id': 'P07', 'cost_cents': 918, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P00', 'cost_cents': 1217, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W06', 'project_id': 'P01', 'cost_cents': 425, 'worker_skill': 'Expert', 'required_skill': 'Intermediate'}, {'worker_id': 'W06', 'project_id': 'P02', 'cost_cents': 1320, 'worker_skill': 'Expert', 'required_skill': 'Senior'}, {'worker_id': 'W06', 'project_id': 'P03', 'cost_cents': 1097, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P04', 'cost_cents': 822, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P05', 'cost_cents': 221, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P06', 'cost_cents': 496, 'worker_skill': 'Expert', 'required_skill': 'Junior'}, {'worker_id': 'W06', 'project_id': 'P07', 'cost_cents': 908, 'worker_skill': 'Expert', 'required_skill': 'Junior'}]
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
workers = ['W10', 'W11', 'W20', 'W14', 'W15', 'W06']
skill_level = {'Junior': 0, 'Intermediate': 1, 'Senior': 2, 'Expert': 3}
A = set()
cost = {}
for offer in offers:
    w = offer['worker_id']
    p = offer['project_id']
    ws = offer['worker_skill']
    rs = offer['required_skill']
    if skill_level[ws] >= skill_level[rs]:
        A.add((w, p))
        if w not in cost:
            cost[w] = {}
        cost[w][p] = offer['cost_cents']
for p in projects:
    if not any(((w, p) in A for w in workers)):
        raise ValueError(f'No eligible offers for project {p}')
for w in workers:
    pass
m = gp.Model('assignment')
x_vars = m.addVars(A, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in A)), GRB.MINIMIZE)
for p in projects:
    m.addConstr(gp.quicksum((x_vars[w, p] for w in workers if (w, p) in A)) == 1, name='prj_' + p)
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, p] for p in projects if (w, p) in A)) <= 1, name='wrk_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')