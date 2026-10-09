import gurobipy as gp
from gurobipy import GRB
workers = ['W00', 'W04', 'W06', 'W10', 'W11', 'W14', 'W15', 'W19', 'W20']
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
offers = {('W00', 'P04'): 430, ('W00', 'P05'): 1385, ('W00', 'P06'): 260, ('W00', 'P07'): 1107, ('W04', 'P03'): 543, ('W04', 'P04'): 205, ('W04', 'P05'): 1149, ('W04', 'P06'): 533, ('W04', 'P07'): 732, ('W06', 'P00'): 1217, ('W06', 'P01'): 425, ('W06', 'P02'): 1320, ('W06', 'P03'): 1097, ('W06', 'P04'): 822, ('W06', 'P05'): 221, ('W06', 'P06'): 496, ('W06', 'P07'): 908, ('W10', 'P00'): 147, ('W10', 'P01'): 114, ('W10', 'P03'): 998, ('W10', 'P04'): 1270, ('W10', 'P07'): 953, ('W11', 'P00'): 235, ('W11', 'P02'): 1034, ('W11', 'P03'): 782, ('W11', 'P04'): 556, ('W11', 'P05'): 1018, ('W11', 'P07'): 1136, ('W14', 'P01'): 1361, ('W14', 'P03'): 379, ('W14', 'P04'): 239, ('W14', 'P05'): 460, ('W14', 'P06'): 130, ('W15', 'P00'): 722, ('W15', 'P02'): 577, ('W15', 'P03'): 1062, ('W15', 'P04'): 1314, ('W15', 'P05'): 528, ('W15', 'P06'): 835, ('W15', 'P07'): 918, ('W19', 'P03'): 109, ('W19', 'P04'): 1306, ('W19', 'P05'): 626, ('W19', 'P06'): 466, ('W19', 'P07'): 1365, ('W20', 'P00'): 675, ('W20', 'P01'): 1055, ('W20', 'P03'): 703, ('W20', 'P06'): 893, ('W20', 'P07'): 129}
project2workers = {p: [] for p in projects}
for (w, p) in offers:
    project2workers[p].append(w)
worker2projects = {w: [] for w in workers}
for (w, p) in offers:
    worker2projects[w].append(p)
for p in projects:
    if len(project2workers[p]) == 0:
        raise ValueError(f'No valid offers for project {p}')
m = gp.Model('worker_project_assignment')
x = m.addVars(offers.keys(), vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((offers[w, p] * x[w, p] for (w, p) in offers)), GRB.MINIMIZE)
for p in projects:
    m.addConstr(gp.quicksum((x[w, p] for w in project2workers[p])) == 1, name='prj_' + p)
for w in workers:
    m.addConstr(gp.quicksum((x[w, p] for p in worker2projects[w])) <= 1, name='wrk_' + w)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')