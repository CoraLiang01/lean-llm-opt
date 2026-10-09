import gurobipy as gp
from gurobipy import GRB
workers = ['W00', 'W04', 'W06', 'W10', 'W11', 'W14', 'W15', 'W19', 'W20']
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
offers = [{'worker_id': 'W10', 'project_id': 'P00', 'cost_cents': 147}, {'worker_id': 'W10', 'project_id': 'P01', 'cost_cents': 114}, {'worker_id': 'W10', 'project_id': 'P03', 'cost_cents': 998}, {'worker_id': 'W10', 'project_id': 'P04', 'cost_cents': 1270}, {'worker_id': 'W10', 'project_id': 'P07', 'cost_cents': 953}, {'worker_id': 'W19', 'project_id': 'P03', 'cost_cents': 109}, {'worker_id': 'W19', 'project_id': 'P04', 'cost_cents': 1306}, {'worker_id': 'W19', 'project_id': 'P05', 'cost_cents': 626}, {'worker_id': 'W19', 'project_id': 'P06', 'cost_cents': 466}, {'worker_id': 'W19', 'project_id': 'P07', 'cost_cents': 1365}, {'worker_id': 'W11', 'project_id': 'P00', 'cost_cents': 235}, {'worker_id': 'W11', 'project_id': 'P02', 'cost_cents': 1034}, {'worker_id': 'W11', 'project_id': 'P03', 'cost_cents': 782}, {'worker_id': 'W11', 'project_id': 'P04', 'cost_cents': 556}, {'worker_id': 'W11', 'project_id': 'P05', 'cost_cents': 1018}, {'worker_id': 'W11', 'project_id': 'P07', 'cost_cents': 1136}, {'worker_id': 'W00', 'project_id': 'P01', 'cost_cents': 17}, {'worker_id': 'W00', 'project_id': 'P04', 'cost_cents': 430}, {'worker_id': 'W00', 'project_id': 'P05', 'cost_cents': 1385}, {'worker_id': 'W00', 'project_id': 'P06', 'cost_cents': 260}, {'worker_id': 'W00', 'project_id': 'P07', 'cost_cents': 1107}, {'worker_id': 'W20', 'project_id': 'P00', 'cost_cents': 675}, {'worker_id': 'W20', 'project_id': 'P01', 'cost_cents': 1055}, {'worker_id': 'W20', 'project_id': 'P03', 'cost_cents': 703}, {'worker_id': 'W20', 'project_id': 'P06', 'cost_cents': 893}, {'worker_id': 'W20', 'project_id': 'P07', 'cost_cents': 129}, {'worker_id': 'W04', 'project_id': 'P03', 'cost_cents': 543}, {'worker_id': 'W04', 'project_id': 'P04', 'cost_cents': 205}, {'worker_id': 'W04', 'project_id': 'P05', 'cost_cents': 1149}, {'worker_id': 'W04', 'project_id': 'P06', 'cost_cents': 533}, {'worker_id': 'W04', 'project_id': 'P07', 'cost_cents': 732}, {'worker_id': 'W06', 'project_id': 'P00', 'cost_cents': 1217}, {'worker_id': 'W06', 'project_id': 'P01', 'cost_cents': 425}, {'worker_id': 'W06', 'project_id': 'P02', 'cost_cents': 1320}, {'worker_id': 'W06', 'project_id': 'P03', 'cost_cents': 1097}, {'worker_id': 'W06', 'project_id': 'P04', 'cost_cents': 822}, {'worker_id': 'W06', 'project_id': 'P05', 'cost_cents': 221}, {'worker_id': 'W06', 'project_id': 'P06', 'cost_cents': 496}, {'worker_id': 'W06', 'project_id': 'P07', 'cost_cents': 908}, {'worker_id': 'W14', 'project_id': 'P01', 'cost_cents': 1361}, {'worker_id': 'W14', 'project_id': 'P03', 'cost_cents': 379}, {'worker_id': 'W14', 'project_id': 'P04', 'cost_cents': 239}, {'worker_id': 'W14', 'project_id': 'P05', 'cost_cents': 460}, {'worker_id': 'W14', 'project_id': 'P06', 'cost_cents': 130}, {'worker_id': 'W15', 'project_id': 'P00', 'cost_cents': 722}, {'worker_id': 'W15', 'project_id': 'P02', 'cost_cents': 577}, {'worker_id': 'W15', 'project_id': 'P03', 'cost_cents': 1062}, {'worker_id': 'W15', 'project_id': 'P04', 'cost_cents': 1314}, {'worker_id': 'W15', 'project_id': 'P05', 'cost_cents': 528}, {'worker_id': 'W15', 'project_id': 'P06', 'cost_cents': 835}, {'worker_id': 'W15', 'project_id': 'P07', 'cost_cents': 918}]
O = set()
cost = {}
for offer in offers:
    w = offer['worker_id']
    p = offer['project_id']
    c = offer['cost_cents']
    O.add((w, p))
    if w not in cost:
        cost[w] = {}
    cost[w][p] = c
for (w, p) in O:
    if w not in cost or p not in cost[w]:
        raise ValueError(f'Missing cost for ({w},{p}) in O')
m = gp.Model('worker_project_assignment')
x_vars = m.addVars(O, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in O)), GRB.MINIMIZE)
for p in projects:
    eligible_workers = [w for w in workers if (w, p) in O]
    m.addConstr(gp.quicksum((x_vars[w, p] for w in eligible_workers)) == 1, name='prj_' + p)
for w in workers:
    eligible_projects = [p for p in projects if (w, p) in O]
    m.addConstr(gp.quicksum((x_vars[w, p] for p in eligible_projects)) <= 1, name='wrk_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')