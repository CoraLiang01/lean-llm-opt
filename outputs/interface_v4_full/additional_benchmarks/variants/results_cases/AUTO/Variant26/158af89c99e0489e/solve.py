import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    workers = ['W00', 'W04', 'W06', 'W10', 'W11', 'W14', 'W15', 'W19', 'W20']
    projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
    eligible_data = [('W06', 'P00', 1217), ('W10', 'P00', 147), ('W11', 'P00', 235), ('W15', 'P00', 722), ('W20', 'P00', 675), ('W06', 'P01', 425), ('W10', 'P01', 114), ('W14', 'P01', 1361), ('W20', 'P01', 1055), ('W06', 'P02', 1320), ('W11', 'P02', 1034), ('W15', 'P02', 577), ('W04', 'P03', 543), ('W06', 'P03', 1097), ('W10', 'P03', 998), ('W11', 'P03', 782), ('W14', 'P03', 379), ('W15', 'P03', 1062), ('W19', 'P03', 109), ('W20', 'P03', 703), ('W00', 'P04', 430), ('W04', 'P04', 205), ('W06', 'P04', 822), ('W10', 'P04', 1270), ('W11', 'P04', 556), ('W14', 'P04', 239), ('W15', 'P04', 1314), ('W19', 'P04', 1306), ('W00', 'P05', 1385), ('W04', 'P05', 1149), ('W06', 'P05', 221), ('W11', 'P05', 1018), ('W14', 'P05', 460), ('W15', 'P05', 528), ('W19', 'P05', 626), ('W00', 'P06', 260), ('W04', 'P06', 533), ('W06', 'P06', 496), ('W14', 'P06', 130), ('W15', 'P06', 835), ('W19', 'P06', 466), ('W20', 'P06', 893), ('W00', 'P07', 1107), ('W04', 'P07', 732), ('W06', 'P07', 908), ('W10', 'P07', 953), ('W11', 'P07', 1136), ('W15', 'P07', 918), ('W19', 'P07', 1365), ('W20', 'P07', 129)]
    eligible = set()
    cost = {}
    for w, p, c in eligible_data:
        eligible.add((w, p))
        if w not in cost:
            cost[w] = {}
        cost[w][p] = c
    for w, p in eligible:
        if w not in cost or p not in cost[w]:
            raise ValueError(f'Missing cost for ({w},{p})')
    m = gp.Model('assignment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars([(w, p) for w, p in eligible], vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w][p] * x[w, p] for w, p in eligible)), GRB.MINIMIZE)
    for p in projects:
        eligible_workers = [w for w in workers if (w, p) in eligible]
        m.addConstr(gp.quicksum((x[w, p] for w in eligible_workers)) == 1, name='prj_' + p)
    for w in workers:
        eligible_projects = [p for p in projects if (w, p) in eligible]
        m.addConstr(gp.quicksum((x[w, p] for p in eligible_projects)) <= 1, name='wrk_' + w)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()